import os
import sys
import argparse
import numpy as np
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import config as cfg
from models.unet import UNetModel
from dataloader import dataloader, get_test_dataloader
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

class Langevin(torch.nn.Module):
    """
    Implémente une trajectoire de sampling par dynamique de Langevin.

    Concept débutant :
    - on part d'un état initial (images ou bruit)
    - on applique plusieurs petites mises à jour
    - à chaque étape, on ajoute un bruit aléatoire
    - le réseau (net) sert à guider ces mises à jour
    """   
     
    def __init__(self, num_steps, shape, gammas, device=None, mean_match=True):
        super().__init__()
        self.mean_match = mean_match
        self.num_steps = num_steps
        self.d = shape
        self.gammas = gammas.float()
        self.device = device if device is not None else gammas.device
        self.time = torch.cumsum(self.gammas, 0).to(self.device).float()

    def sample(self, net, init_samples):
        """
        Effectue l'inférence (sampling) sur init_samples en utilisant le réseau 'net'.
        Retourne la trajectoire complète et l'image finale.
        """
        x = init_samples
        N = x.shape[0]
        steps = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        
        x_tot = torch.zeros(N, self.num_steps, *self.d).to(x.device)

        with torch.no_grad():
            for k in range(self.num_steps):
                gamma = self.gammas[k]

                net_out = net(x, steps[:, k, :])
                
                if self.mean_match:
                    t_old = net_out
                else:
                    t_old = x + net_out

                if k == self.num_steps - 1:
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                
                x_tot[:, k, :] = x
        
        return x_tot, x

def get_model(device):
    """Construit deux réseaux UNet :
    - net_f : réseau "forward"
    - net_b : réseau "backward"

    Concept :
    - le Schrödinger Bridge entraîne deux directions (aller/retour)
    - les deux réseaux apprennent à se "répondre" via IPF.
    """
    image_size = cfg.IMAGE_SIZE
    channel_mult = (1, 1, 2, 2, 4, 4)
    attention_ds = [image_size // int(res) for res in cfg.ATTENTION_RESOLUTIONS.split(",")]
    net = UNetModel(
    in_channels=cfg.CHANNELS,
    model_channels=cfg.NUM_CHANNELS,
    out_channels=cfg.CHANNELS,
    num_res_blocks=cfg.NUM_RES_BLOCKS,
    attention_resolutions=tuple(attention_ds),
    dropout=cfg.DROPOUT,
    channel_mult= channel_mult,
    num_heads=cfg.NUM_HEADS,
    num_heads_upsample=cfg.NUM_HEADS_UPSAMPLE)

    return net.to(device)

def save_results(input_batch, output_batch, output_dir, batch_idx, paired_files=None, cd30_dir=None):
    """
    Sauvegarde HES (input), CD30 virtuel (output), CD30 réel (paired) côte à côte avec légende sous chaque image.
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_size = input_batch.shape[0]
    for i in range(batch_size):
        hes_img = to_pil_image(input_batch[i].cpu())
        cd30_virtual_img = to_pil_image(output_batch[i].cpu())
        cd30_real_img = None
        if paired_files is not None and cd30_dir is not None:
            fname = paired_files[batch_idx * batch_size + i]
            cd30_real_path = os.path.join(cd30_dir, fname)
            cd30_real_img = Image.open(cd30_real_path).convert('RGB')

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(hes_img)
        axes[0].set_title('HES')
        axes[0].axis('off')
        axes[1].imshow(cd30_virtual_img)
        axes[1].set_title('CD30 virtuel')
        axes[1].axis('off')
        if cd30_real_img is None:
            arr = np.zeros((hes_img.size[1], hes_img.size[0], 3), dtype=np.uint8)
        else:
            arr = np.array(cd30_real_img)
            if arr is None or arr.dtype != np.uint8:
                arr = np.zeros((hes_img.size[1], hes_img.size[0], 3), dtype=np.uint8)
        try:
            axes[2].imshow(arr)
        except Exception as e:
            axes[2].imshow(np.zeros((hes_img.size[1], hes_img.size[0], 3), dtype=np.uint8))
            axes[2].set_title(f'CD30 réel (erreur)')
        axes[2].set_title('CD30 réel')
        axes[2].axis('off')

        for ax in axes:
            ax.set_xlabel('')
        plt.tight_layout()
        out_name = f"{output_dir}/pred_{batch_idx}_{i}.png"
        plt.savefig(out_name)
        plt.close(fig)

def run_inference(ckpt_path, output_dir='./results'):
    device = torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    net = get_model(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()

    n = cfg.NUM_STEPS // 2
    gamma_half = np.linspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n) if cfg.GAMMA_SPACE == 'linspace' else np.geomspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
    gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
    gammas = torch.tensor(gammas).to(device)
    
    langevin = Langevin(cfg.NUM_STEPS, (cfg.CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), gammas, device=device, mean_match=cfg.MEAN_MATCH)

    test_loader = get_test_dataloader()
    test_dataset = test_loader.dataset
    paired_files = test_dataset.paired_files
    cd30_dir = test_dataset.cd30_dir

    print("Starting generation...")
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            # data = (img, fname) par batch
            if isinstance(data, tuple) or isinstance(data, list):
                batch = data[0].to(device)
            else:
                batch = data.to(device)
            _, final_image = langevin.sample(net, batch)
            save_results(batch, final_image, output_dir, i, paired_files, cd30_dir)

checkpoint = 'checkpoints/net_f_1.ckpt' # Exemple de chemin vers un checkpoint
out_dir = 'results' # Exemple de répertoire de sortie

run_inference(checkpoint, out_dir)