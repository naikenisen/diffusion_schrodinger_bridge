import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader

import config as cfg
from dataloader import get_datasets
from models.unet import UNetModel
from models.utils import Langevin, CacheLoader, EMAHelper, repeater

class IPFTrainer:
    def __init__(self):
        self.accelerator = Accelerator(mixed_precision="fp16")
        self.device = self.accelerator.device
        
        # Configuration
        self.n_ipf = cfg.N_IPF
        self.num_steps = cfg.NUM_STEPS
        self.batch_size = cfg.BATCH_SIZE
        
        # Time schedule
        n = self.num_steps // 2
        if cfg.GAMMA_SPACE == 'linspace':
            gamma_half = np.linspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
        else:
            gamma_half = np.geomspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
        gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        self.gammas = torch.tensor(gammas).to(self.device)
        self.T = torch.sum(self.gammas)

        # Models
        self.nets = torch.nn.ModuleDict({
            'f': self._build_unet(),
            'b': self._build_unet()
        })

        # Optimizers
        self.opts = {
            'f': torch.optim.Adam(self.nets['f'].parameters(), lr=cfg.LR),
            'b': torch.optim.Adam(self.nets['b'].parameters(), lr=cfg.LR)
        }

        # EMA
        self.emas = {
            'f': EMAHelper(mu=cfg.EMA_RATE, device=self.device),
            'b': EMAHelper(mu=cfg.EMA_RATE, device=self.device)
        }
        self.emas['f'].register(self.nets['f'])
        self.emas['b'].register(self.nets['b'])

        # Data & Langevin
        init_ds, final_ds, mean_final, var_final = get_datasets()
        self.mean_final = mean_final.to(self.device)
        self.var_final = var_final.to(self.device)
        self.std_final = torch.sqrt(var_final).to(self.device)

        dl_args = {"num_workers": cfg.NUM_WORKERS, "drop_last": True, "shuffle": True}
        self.dl_init = repeater(self.accelerator.prepare(DataLoader(init_ds, batch_size=cfg.CACHE_NPAR, **dl_args)))
        self.dl_final = repeater(self.accelerator.prepare(DataLoader(final_ds, batch_size=cfg.CACHE_NPAR, **dl_args)))

        # Prepare everything with Accelerator
        self.nets['f'], self.opts['f'] = self.accelerator.prepare(self.nets['f'], self.opts['f'])
        self.nets['b'], self.opts['b'] = self.accelerator.prepare(self.nets['b'], self.opts['b'])

        # Langevin setup
        dummy = next(self.dl_init)[0]
        self.langevin = Langevin(
            self.num_steps, dummy.shape[1:], self.gammas, 
            device=self.device, mean_final=self.mean_final, var_final=self.var_final
        )

        os.makedirs('./checkpoints', exist_ok=True)

    def _build_unet(self):
        att_ds = tuple(256 // int(res) for res in cfg.ATTENTION_RESOLUTIONS.split(","))
        return UNetModel(
            model_channels=cfg.NUM_CHANNELS,
            num_res_blocks=cfg.NUM_RES_BLOCKS,
            attention_resolutions=att_ds,
            dropout=cfg.DROPOUT,
            channel_mult=(1, 1, 2, 2, 4, 4)
        )

    def get_cache_loader(self, direction, n):
        # Determine source direction and settings
        is_b = (direction == 'b')
        src_dir = 'f' if is_b else 'b'

        # Get EMA sample net
        sample_net = self.emas[src_dir].ema_copy(self.nets[src_dir])

        loader = CacheLoader(
            direction, sample_net,
            dataloader_b=self.dl_init,
            dataloader_f=self.dl_final,
            num_batches=cfg.NUM_CACHE_BATCHES,
            langevin=self.langevin,
            n=n,
            mean=None if is_b else self.mean_final,
            std=None if is_b else self.std_final,
            batch_size=cfg.CACHE_NPAR,
            device=self.device
        )

        return repeater(self.accelerator.prepare(DataLoader(loader, batch_size=self.batch_size, num_workers=0)))

    def save(self, direction, n):
        if self.accelerator.is_local_main_process:
            path = f'./checkpoints/net_{direction}_{n}.ckpt'
            model = self.emas[direction].ema_copy(self.nets[direction])
            torch.save(model.state_dict(), path)

    def ipf_step(self, direction, n):
        loader = self.get_cache_loader(direction, n)
        
        pbar = tqdm(range(cfg.NUM_ITER), disable=not self.accelerator.is_local_main_process)
        for i in pbar:
            # Refresh cache periodically
            if i > 0 and i % cfg.CACHE_REFRESH_STRIDE == 0:
                loader = self.get_cache_loader(direction, n)
            
            x, out, steps = next(loader)
            
            # Forward pass
            # Note: steps are flipped for network input vs langevin recording
            t = self.T - steps.to(self.device)
            pred = self.nets[direction](x, t) - x
            
            loss = F.mse_loss(pred, out)
            
            # Optimization
            self.opts[direction].zero_grad()
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.nets[direction].parameters(), cfg.GRAD_CLIP)
            self.opts[direction].step()
            
            self.emas[direction].update(self.nets[direction])
            
            pbar.set_description(f"IPF {n} ({direction}) | Loss: {loss.item():.4f}")

        self.save(direction, n)

    def train(self):
        print("Training Started...")
        for n in range(1, self.n_ipf + 1):
            self.ipf_step('b', n)
            self.ipf_step('f', n)

if __name__ == "__main__":
    IPFTrainer().train()