import copy
from itertools import repeat
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

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
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            inner_module = module.module
            module_copy = copy.deepcopy(inner_module).to(self.device)
            if isinstance(module, nn.DataParallel):
                module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = copy.deepcopy(module).to(self.device)

        for name, param in module_copy.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)
        return module_copy

class Langevin(nn.Module):
    def __init__(self, num_steps, shape, gammas, device=None,
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

    def get_gaussian_grad(self, x):
        return -(x - self.mean_final.to(x.device)) / self.var_final.to(x.device)

    def record_init_langevin(self, init_samples):
        x = init_samples
        N = x.shape[0]
        x_tot = torch.zeros(N, self.num_steps, *self.d, device=x.device)
        out = torch.zeros(N, self.num_steps, *self.d, device=x.device)
        steps_expanded = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))

        for k in range(self.num_steps):
            gamma = self.gammas[k]
            gradx = self.get_gaussian_grad(x)
            t_old = x + gamma * gradx
            z = torch.randn_like(x)
            x = t_old + torch.sqrt(2 * gamma) * z
            
            gradx_new = self.get_gaussian_grad(x)
            t_new = x + gamma * gradx_new
            
            x_tot[:, k] = x
            out[:, k] = (t_old - t_new)

        return x_tot, out, steps_expanded

    def record_langevin_seq(self, net, init_samples, ipf_it=0):
        x = init_samples
        N = x.shape[0]
        steps = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        
        x_tot = torch.zeros(N, self.num_steps, *self.d, device=x.device)
        out = torch.zeros(N, self.num_steps, *self.d, device=x.device)

        for k in range(self.num_steps):
            gamma = self.gammas[k]
            net_out = net(x, steps[:, k])
            
            t_old = net_out if self.mean_match else x + net_out
            
            z = torch.randn_like(x)
            x = t_old + torch.sqrt(2 * gamma) * z
            
            net_out_new = net(x, steps[:, k])
            
            t_new = net_out_new if self.mean_match else x + net_out_new

            x_tot[:, k] = x
            out[:, k] = (t_old - t_new)

        return x_tot, out, steps

class CacheLoader(Dataset):
    def __init__(self, fb, sample_net, dataloader_b, num_batches, langevin, n,
                 mean, std, batch_size, dataloader_f, device='cpu'):
        super().__init__()
        shape = langevin.d
        
        data_list = []
        steps_list = []

        with torch.no_grad():
            for b in tqdm(range(num_batches)):
                if fb == 'b':
                    batch = next(dataloader_b)[0].to(device)
                elif fb == 'f':
                    batch = next(dataloader_f)[0].to(device)
                else:
                    batch = mean + std * torch.randn((batch_size, *shape), device=device)

                if (n == 1) and (fb == 'b'):
                    x, out, steps_expanded = langevin.record_init_langevin(batch)
                else:
                    x, out, steps_expanded = langevin.record_langevin_seq(sample_net, batch, ipf_it=n)

                x = x.cpu().unsqueeze(2)
                out = out.cpu().unsqueeze(2)
                batch_data = torch.cat((x, out), dim=2)

                data_list.append(batch_data.flatten(start_dim=0, end_dim=1))
                steps_list.append(steps_expanded.cpu().flatten(start_dim=0, end_dim=1))

        self.data = torch.cat(data_list, dim=0)
        self.steps_data = torch.cat(steps_list, dim=0)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], self.steps_data[index]

    def __len__(self):
        return self.data.shape[0]