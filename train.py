import os
import sys
import time
import random
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import torchvision.transforms as transforms
from tqdm import tqdm
from accelerate import Accelerator
from pytorch_lightning.loggers import CSVLogger as _CSVLogger

# Imports locaux supposés existants
import config as cfg
from models.unet import UNetModel
from dataloader import dataloader

cmp = lambda x: transforms.Compose([*x])

# ============================================================================
#  UTILS & CLASSES (Nécessaires pour l'entraînement)
# ============================================================================

def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

class EMAHelper:
    def __init__(self, mu=0.999, device="cpu"):
        self.mu = mu
        self.shadow = {}
        self.device = device

    def register(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema_copy(self, module):
        # Crée une copie du modèle avec les poids lissés (EMA)
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            inner_module = module.module
            locs = inner_module.locals
            module_copy = type(inner_module)(*locs).to(self.device)
            module_copy.load_state_dict(inner_module.state_dict())
            if isinstance(module, nn.DataParallel):
                module_copy = nn.DataParallel(module_copy)
        else:
            locs = module.locals
            module_copy = type(module)(*locs).to(self.device)
            module_copy.load_state_dict(module.state_dict())
        
        # Charge les poids EMA dans la copie
        for name, param in module_copy.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)
        return module_copy

# ============================================================================
#  LANGEVIN DYNAMICS (Cœur de la physique du modèle)
# ============================================================================

def grad_gauss(x, m, var):
    return -(x - m) / var

class Langevin(torch.nn.Module):
    def __init__(self, num_steps, shape, gammas, time_sampler, device=None,
                 mean_final=torch.tensor([0., 0.]), var_final=torch.tensor([.5, .5]),
                 mean_match=True):
        super().__init__()
        self.mean_match = mean_match
        self.mean_final = mean_final
        self.var_final = var_final
        self.num_steps = num_steps
        self.d = shape
        self.gammas = gammas.float()
        self.device = device if device is not None else gammas.device
        self.time = torch.cumsum(self.gammas, 0).to(self.device).float()

    def record_init_langevin(self, init_samples):
        # Diffusion initiale (bruit gaussien)
        x = init_samples
        N = x.shape[0]
        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        steps_expanded = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))

        for k in range(self.num_steps):
            gamma = self.gammas[k]
            gradx = grad_gauss(x, self.mean_final, self.var_final)
            t_old = x + gamma * gradx
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma) * z
            
            gradx_new = grad_gauss(x, self.mean_final, self.var_final)
            t_new = x + gamma * gradx_new
            
            x_tot[:, k, :] = x
            out[:, k, :] = (t_old - t_new)

        return x_tot, out, steps_expanded

    def record_langevin_seq(self, net, init_samples, ipf_it=0):
        # Diffusion guidée par le réseau de neurones
        x = init_samples
        N = x.shape[0]
        steps = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        
        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)

        for k in range(self.num_steps):
            gamma = self.gammas[k]
            # Prediction du réseau
            net_out = net(x, steps[:, k, :])
            
            if self.mean_match:
                t_old = net_out
            else:
                t_old = x + net_out

            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma) * z
            
            # Recalcul pour la target (training objective)
            net_out_new = net(x, steps[:, k, :])
            if self.mean_match:
                t_new = net_out_new
            else:
                t_new = x + net_out_new

            x_tot[:, k, :] = x
            out[:, k, :] = (t_old - t_new)

        return x_tot, out, steps

# ============================================================================
#  CACHE LOADER (Pour optimiser l'IPF)
# ============================================================================

class CacheLoader(Dataset):
    def __init__(self, fb, sample_net, dataloader_b, num_batches, langevin, n,
                 mean, std, batch_size, device='cpu', dataloader_f=None, transfer=False):
        super().__init__()
        shape = langevin.d
        num_steps = langevin.num_steps
        self.data = torch.zeros((num_batches, batch_size * num_steps, 2, *shape))
        self.steps_data = torch.zeros((num_batches, batch_size * num_steps, 1))

        print(f"Generating cache for iteration {n} ({fb})...")
        with torch.no_grad():
            for b in tqdm(range(num_batches)):
                if fb == 'b':
                    batch = next(dataloader_b)[0].to(device)
                elif fb == 'f' and transfer:
                    batch = next(dataloader_f)[0].to(device)
                else:
                    batch = mean + std * torch.randn((batch_size, *shape), device=device)

                if (n == 1) and (fb == 'b'):
                    x, out, steps_expanded = langevin.record_init_langevin(batch)
                else:
                    x, out, steps_expanded = langevin.record_langevin_seq(sample_net, batch, ipf_it=n)

                # Stockage CPU
                x = x.cpu().unsqueeze(2)
                out = out.cpu().unsqueeze(2)
                batch_data = torch.cat((x, out), dim=2)
                
                self.data[b] = batch_data.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = steps_expanded.cpu().flatten(start_dim=0, end_dim=1)
                
                del x, out, batch_data
                torch.cuda.empty_cache()

        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], self.steps_data[index]

    def __len__(self):
        return self.data.shape[0]

# ============================================================================
#  SETUP FUNCTIONS
# ============================================================================

def get_models():
    # Configuration U-Net simplifiée basée sur cfg
    image_size = cfg.IMAGE_SIZE
    # (Logique de channel_mult identique à l'original)
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
    return UNetModel(**kwargs), UNetModel(**kwargs)

def get_datasets():
    # Chargement dataset TRAIN
    train_transform = [
        transforms.Resize(cfg.IMAGE_SIZE), transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor()
    ]
    if cfg.RANDOM_FLIP: train_transform.insert(2, transforms.RandomHorizontalFlip())
    
    root = os.path.join(cfg.DATA_DIR, 'dataset_v4')
    # On charge les paires HES/CD30 du split 'train'
    init_ds = dataloader(root, image_size=cfg.IMAGE_SIZE, transform=cmp(train_transform), split='train')
    final_ds = dataloader(root, image_size=cfg.IMAGE_SIZE, transform=cmp(train_transform), split='train')
    mean_final = torch.tensor(0.)
    var_final = torch.tensor(1. * 10 ** 3)
    return init_ds, final_ds, mean_final, var_final

# ============================================================================
#  MAIN TRAINER CLASS
# ============================================================================

class IPFTrainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.accelerator = Accelerator(mixed_precision="no", cpu=(cfg.DEVICE == 'cpu'))
        self.device = self.accelerator.device
        
        # Params
        self.n_ipf = cfg.N_IPF
        self.num_steps = cfg.NUM_STEPS
        self.batch_size = cfg.BATCH_SIZE
        self.lr = cfg.LR
        
        # Gammas
        n = self.num_steps // 2
        gamma_half = np.linspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n) if cfg.GAMMA_SPACE == 'linspace' else np.geomspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
        gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        gammas = torch.tensor(gammas).to(self.device)
        self.T = torch.sum(gammas)

        # Models & Optimizer
        self.net = nn.ModuleDict()
        net_f, net_b = get_models()
        self.net['f'] = net_f.to(self.device)
        self.net['b'] = net_b.to(self.device)
        
        self.optimizer = {
            'f': torch.optim.Adam(self.net['f'].parameters(), lr=self.lr),
            'b': torch.optim.Adam(self.net['b'].parameters(), lr=self.lr)
        }

        # EMA
        if cfg.EMA:
            self.ema_helpers = {
                'f': EMAHelper(mu=cfg.EMA_RATE, device=self.device),
                'b': EMAHelper(mu=cfg.EMA_RATE, device=self.device)
            }
            self.ema_helpers['f'].register(self.net['f'])
            self.ema_helpers['b'].register(self.net['b'])

        # Data
        init_ds, final_ds, mean_final, var_final = get_datasets()
        self.mean_final = mean_final.to(self.device)
        self.var_final = var_final.to(self.device)
        self.std_final = torch.sqrt(var_final).to(self.device)
        
        # Dataloaders (Training)
        dl_kwargs = {"num_workers": cfg.NUM_WORKERS, "pin_memory": True, "drop_last": True, "shuffle": True}
        self.cache_init_dl = repeater(self.accelerator.prepare(DataLoader(init_ds, batch_size=cfg.CACHE_NPAR, **dl_kwargs)))
        
        if cfg.TRANSFER:
            self.cache_final_dl = repeater(self.accelerator.prepare(DataLoader(final_ds, batch_size=cfg.CACHE_NPAR, **dl_kwargs)))
        else:
            self.cache_final_dl = None

        # Langevin
        prob_vec = gammas * 0 + 1 # Simplification distribution uniforme si non spécifié
        time_sampler = torch.distributions.categorical.Categorical(prob_vec)
        # On récupère la shape depuis un batch
        dummy_batch = next(self.cache_init_dl)[0]
        self.langevin = Langevin(self.num_steps, dummy_batch.shape[1:], gammas, time_sampler, 
                                 device=self.device, mean_final=self.mean_final, var_final=self.var_final, 
                                 mean_match=cfg.MEAN_MATCH)
        
        # Logging
        self.logger = _CSVLogger(cfg.CSV_LOG_DIR, name='logs')
        os.makedirs('./checkpoints', exist_ok=True)

    def new_cacheloader(self, forward_or_backward, n):
        # Determine sample network direction (opposite of training direction)
        sample_dir = 'f' if forward_or_backward == 'b' else 'b'
        
        # Use EMA for sampling if enabled
        if cfg.EMA:
            sample_net = self.ema_helpers[sample_dir].ema_copy(self.net[sample_dir])
        else:
            sample_net = self.net[sample_dir]

        # Configurer le cache loader
        if forward_or_backward == 'b':
            # Training Backward: Sample from Forward net starting from Init distribution
            dl = CacheLoader('b', sample_net, self.cache_init_dl, cfg.NUM_CACHE_BATCHES, 
                             self.langevin, n, mean=None, std=None, batch_size=cfg.CACHE_NPAR, 
                             device=self.device, dataloader_f=self.cache_final_dl, transfer=cfg.TRANSFER)
        else:
            # Training Forward: Sample from Backward net (or Prior)
            dl = CacheLoader('f', sample_net, None, cfg.NUM_CACHE_BATCHES, 
                             self.langevin, n, mean=self.mean_final, std=self.std_final, batch_size=cfg.CACHE_NPAR, 
                             device=self.device, dataloader_f=self.cache_final_dl, transfer=cfg.TRANSFER)
        
        return repeater(self.accelerator.prepare(DataLoader(dl, batch_size=self.batch_size)))

    def save_checkpoint(self, fb, n, i):
        if self.accelerator.is_local_main_process:
            filename = f'./checkpoints/net_{fb}_{n}.ckpt'
            # Sauvegarder l'EMA si activé car c'est mieux pour l'inférence
            if cfg.EMA:
                print(f"Saving EMA weights to {filename}")
                model_to_save = self.ema_helpers[fb].ema_copy(self.net[fb])
                torch.save(model_to_save.state_dict(), filename)
            else:
                print(f"Saving Standard weights to {filename}")
                torch.save(self.net[fb].state_dict(), filename)

    def ipf_step(self, fb, n):
        print(f"Starting IPF step {n} for direction {fb}")
        train_dl = self.new_cacheloader(fb, n)
        self.net[fb] = self.accelerator.prepare(self.net[fb])
        self.optimizer[fb] = self.accelerator.prepare(self.optimizer[fb])

        for i in tqdm(range(cfg.NUM_ITER)):
            x, out, steps_expanded = next(train_dl)
            
            eval_steps = self.T - steps_expanded.to(self.device)
            x = x.to(self.device)
            out = out.to(self.device)

            # Forward pass
            if cfg.MEAN_MATCH:
                pred = self.net[fb](x, eval_steps) - x
            else:
                pred = self.net[fb](x, eval_steps)

            loss = F.mse_loss(pred, out)

            # Backward pass
            self.accelerator.backward(loss)
            
            if cfg.GRAD_CLIPPING:
                torch.nn.utils.clip_grad_norm_(self.net[fb].parameters(), cfg.GRAD_CLIP)

            self.optimizer[fb].step()
            self.optimizer[fb].zero_grad()

            if cfg.EMA:
                self.ema_helpers[fb].update(self.net[fb])
            
            # Log
            if i % cfg.LOG_STRIDE == 0 and self.accelerator.is_local_main_process:
                self.logger.log_metrics({'loss': loss.item(), 'step': i, 'ipf': n, 'dir': fb})

            # Refresh Cache Periodically
            if i > 0 and i % cfg.CACHE_REFRESH_STRIDE == 0:
                train_dl = self.new_cacheloader(fb, n)

        # Fin de l'étape IPF: Sauvegarde
        self.save_checkpoint(fb, n, cfg.NUM_ITER)
        self.logger.save()

    def train(self):
        for n in range(1, self.n_ipf + 1):
            # Alternance Backward / Forward (Algorithme DSB)
            self.ipf_step('b', n)
            self.ipf_step('f', n)

if __name__ == '__main__':
    print('=== DSB Training ===')
    trainer = IPFTrainer()
    trainer.train()