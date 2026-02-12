import os
import sys
import argparse
import numpy as np
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import config as cfg
from models.unet import UNetModel
from dataloader import dataloader
# il faudrait produire des image HES, CD30 virtual et et CD30 real + ecrire legende deesous image avec matplotlib
# revérifier si il va bien infer. py va bien de HES en entrée  vers CD30 en sortie 

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
    if image_size == 256: channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64: channel_mult = (1, 2, 3, 4)
    elif image_size == 32: channel_mult = (1, 2, 2, 2)
    else: raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = [image_size // int(res) for res in cfg.ATTENTION_RESOLUTIONS.split(",")]

    kwargs = {
        "in_channels": cfg.CHANNELS, "model_channels": cfg.NUM_CHANNELS,
        "out_channels": cfg.CHANNELS, "num_res_blocks": cfg.NUM_RES_BLOCKS,
        "attention_resolutions": tuple(attention_ds), "dropout": cfg.DROPOUT,
        "channel_mult": channel_mult, "use_checkpoint": cfg.USE_CHECKPOINT,
        "num_heads": cfg.NUM_HEADS, "num_heads_upsample": cfg.NUM_HEADS_UPSAMPLE,
        "use_scale_shift_norm": cfg.USE_SCALE_SHIFT_NORM,
    }
    net = UNetModel(**kwargs)
    return net.to(device)

def get_test_dataloader():
    """Charge le Dataset de TEST spécifiquement"""
    test_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
    root = os.path.join(cfg.DATA_DIR, 'dataset_v4')
    test_ds = dataloader(root, image_size=cfg.IMAGE_SIZE, domain='HES', transform=test_transform)

    loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    return loader

def save_results(input_batch, output_batch, output_dir, batch_idx):
    """Sauvegarde les images input vs output"""
    comparison = torch.cat([input_batch, output_batch], dim=3)
    vutils.save_image(comparison, f"{output_dir}/pred_{batch_idx}.png", nrow=4, normalize=False)

def run_inference(ckpt_path, output_dir='./results'):
    device = torch.device(cfg.DEVICE)
    print(f"Running inference on {device} using {ckpt_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    net = get_model(device)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    net.eval()

    n = cfg.NUM_STEPS // 2
    gamma_half = np.linspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n) if cfg.GAMMA_SPACE == 'linspace' else np.geomspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
    gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
    gammas = torch.tensor(gammas).to(device)
    
    langevin = Langevin(cfg.NUM_STEPS, (cfg.CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), gammas, device=device, mean_match=cfg.MEAN_MATCH)

    test_loader = get_test_dataloader()
    print(f"Test dataset size: {len(test_loader.dataset)}")

    print("Starting generation...")
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            if isinstance(data, list):
                batch = data[0].to(device)
            else:
                batch = data.to(device)
            _, final_image = langevin.sample(net, batch)
            save_results(batch, final_image, output_dir, i)

    print(f"Inference done. Results saved in {output_dir}")

checkpoint = './checkpoints/bridge_epoch_10.ckpt' # Exemple de chemin vers un checkpoint
out_dir = './results' # Exemple de répertoire de sortie

run_inference(checkpoint, out_dir)