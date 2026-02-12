import os
import sys
import time
import random
import datetime
import numpy as np
from itertools import repeat
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import torchvision.transforms as transforms
from tqdm import tqdm
from accelerate import Accelerator
from pytorch_lightning.loggers import CSVLogger as _CSVLogger
import config as cfg
from models.unet import UNetModel
from dataloader import dataloader, get_datasets
from models.utils import Langevin, CacheLoader, EMAHelper

cmp = lambda x: transforms.Compose([*x])

def get_models():
    image_size = cfg.IMAGE_SIZE
    if image_size == 256: channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64: channel_mult = (1, 2, 3, 4)
    elif image_size == 32: channel_mult = (1, 2, 2, 2)
    else: raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = [image_size // int(res) for res in cfg.ATTENTION_RESOLUTIONS.split(",")]

    net_f = UNetModel(
        in_channels=cfg.CHANNELS,
        model_channels=cfg.NUM_CHANNELS,
        out_channels=cfg.CHANNELS,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        attention_resolutions=tuple(attention_ds),
        dropout=cfg.DROPOUT,
        channel_mult=channel_mult,
        num_heads=cfg.NUM_HEADS,
        num_heads_upsample=cfg.NUM_HEADS_UPSAMPLE
    )
    net_b = UNetModel(
        in_channels=cfg.CHANNELS,
        model_channels=cfg.NUM_CHANNELS,
        out_channels=cfg.CHANNELS,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        attention_resolutions=tuple(attention_ds),
        dropout=cfg.DROPOUT,
        channel_mult=channel_mult,
        num_heads=cfg.NUM_HEADS,
        num_heads_upsample=cfg.NUM_HEADS_UPSAMPLE
    )
    
    # NOTE: Pas de conversion .half() ici, Accelerate s'en charge.
    return net_f, net_b

class IPFTrainer(torch.nn.Module):
    def __init__(self, transfer=True):
        super().__init__()
        # 1. On garde mixed_precision="fp16" ici, c'est lui le chef.
        self.accelerator = Accelerator(mixed_precision="fp16", cpu=False)
        self.device = self.accelerator.device
        self.n_ipf = cfg.N_IPF
        self.num_steps = cfg.NUM_STEPS
        self.batch_size = cfg.BATCH_SIZE
        self.lr = cfg.LR

        self.transfer = transfer


        n = self.num_steps // 2
        gamma_half = np.linspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n) if cfg.GAMMA_SPACE == 'linspace' else np.geomspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
        gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        gammas = torch.tensor(gammas).to(self.device)
        self.T = torch.sum(gammas)
        self.net = nn.ModuleDict()
        net_f, net_b = get_models()
        self.net['f'] = net_f.to(self.device)
        self.net['b'] = net_b.to(self.device)
        self.optimizer = {
            'f': torch.optim.Adam(self.net['f'].parameters(), lr=self.lr),
            'b': torch.optim.Adam(self.net['b'].parameters(), lr=self.lr)
        }
        # Préparer les optimizers une seule fois avec accelerate
        self.net['f'] = self.accelerator.prepare(self.net['f'])
        self.net['b'] = self.accelerator.prepare(self.net['b'])
        self.optimizer['f'] = self.accelerator.prepare(self.optimizer['f'])
        self.optimizer['b'] = self.accelerator.prepare(self.optimizer['b'])
        self.ema_helpers = {
            'f': EMAHelper(mu=cfg.EMA_RATE, device=self.device),
            'b': EMAHelper(mu=cfg.EMA_RATE, device=self.device)
        }
        # Note: Si tu utilises accelerate, l'accès aux paramètres peut nécessiter .module si c'est wrappé
        # Mais EMAHelper gère déjà DataParallel, donc ça devrait aller.
        self.ema_helpers['f'].register(self.net['f'])
        self.ema_helpers['b'].register(self.net['b'])

        init_ds, final_ds, mean_final, var_final = get_datasets()
        self.mean_final = mean_final.to(self.device)
        self.var_final = var_final.to(self.device)
        self.std_final = torch.sqrt(var_final).to(self.device)

        dl_kwargs = {"num_workers": cfg.NUM_WORKERS, "drop_last": True, "shuffle": True}
        self.cache_init_dl = repeater(self.accelerator.prepare(DataLoader(init_ds, batch_size=cfg.CACHE_NPAR, **dl_kwargs)))
        self.cache_final_dl = repeater(self.accelerator.prepare(DataLoader(final_ds, batch_size=cfg.CACHE_NPAR, **dl_kwargs))) if self.transfer else None

        prob_vec = gammas * 0 + 1
        time_sampler = torch.distributions.categorical.Categorical(prob_vec)
        dummy_batch = next(self.cache_init_dl)[0]
        self.langevin = Langevin(self.num_steps, dummy_batch.shape[1:], gammas, time_sampler, 
                                device=self.device, mean_final=self.mean_final, var_final=self.var_final, 
                                mean_match=cfg.MEAN_MATCH)
        os.makedirs('./checkpoints', exist_ok=True)

    def new_cacheloader(self, forward_or_backward, n, transfer=None):
        sample_dir = 'f' if forward_or_backward == 'b' else 'b'
        sample_net = self.ema_helpers[sample_dir].ema_copy(self.net[sample_dir])
        if transfer is None:
            transfer = self.transfer
        if forward_or_backward == 'b':
            dl = CacheLoader('b', sample_net, self.cache_init_dl, cfg.NUM_CACHE_BATCHES, 
                             self.langevin, n, mean=None, std=None, batch_size=cfg.CACHE_NPAR, 
                             device=self.device, dataloader_f=self.cache_final_dl, transfer=transfer)
        else:
            dl = CacheLoader('f', sample_net, None, cfg.NUM_CACHE_BATCHES, 
                             self.langevin, n, mean=self.mean_final, std=self.std_final, batch_size=cfg.CACHE_NPAR, 
                             device=self.device, dataloader_f=self.cache_final_dl, transfer=transfer)
        return repeater(self.accelerator.prepare(DataLoader(
            dl, 
            batch_size=self.batch_size, 
            num_workers=0
        )))

    def save_checkpoint(self, fb, n, i):
        if self.accelerator.is_local_main_process:
            filename = f'./checkpoints/net_{fb}_{n}.ckpt'

            print(f"Saving EMA weights to {filename}")
            model_to_save = self.ema_helpers[fb].ema_copy(self.net[fb])
            torch.save(model_to_save.state_dict(), filename)

    def ipf_step(self, fb, n):
            print(f"Starting IPF step {n} for direction {fb}")
            train_dl = self.new_cacheloader(fb, n)

            import time as _time
            for i in tqdm(range(cfg.NUM_ITER)):
                # ... (chargement des données identique) ...
                x, out, steps_expanded = next(train_dl)
                eval_steps = self.T - steps_expanded.to(self.device)
                x = x.to(self.device)
                out = out.to(self.device)

                # Forward
                # Accelerate gère l'autocast ici si initialisé avec mixed_precision='fp16'
                # Mais le contexte explicite ne fait pas de mal dans les boucles complexes.
                with self.accelerator.autocast():
                    if cfg.MEAN_MATCH:
                        pred = self.net[fb](x, eval_steps) - x
                    else:
                        pred = self.net[fb](x, eval_steps)
                    loss = F.mse_loss(pred, out)
                
                # Backward
                self.accelerator.backward(loss)
                
                # Clipping & Step (SIMPLIFIÉ)
                self.accelerator.clip_grad_norm_(self.net[fb].parameters(), cfg.GRAD_CLIP)

                self.optimizer[fb].step()
                self.optimizer[fb].zero_grad()

                self.ema_helpers[fb].update(self.net[fb])

                # Refresh du cache
                if i > 0 and i % cfg.CACHE_REFRESH_STRIDE == 0:
                    train_dl = self.new_cacheloader(fb, n)
            
            self.save_checkpoint(fb, n, cfg.NUM_ITER)

    def train(self):
        for n in range(1, self.n_ipf + 1):
            self.ipf_step('b', n)
            self.ipf_step('f', n)

print('debut entrainement')
trainer = IPFTrainer()
trainer.train()