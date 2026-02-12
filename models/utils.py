
def repeater(data_loader):
    """
    Concept :
    - Un DataLoader PyTorch s'arrête quand il a tout parcouru.
    - Ici on veut pouvoir appeler next(...) indéfiniment sans gérer les fins d'epoch.
    → On crée une boucle infinie qui répète le DataLoader.
    """
    for loader in repeat(data_loader):
        for data in loader:
            yield data

class EMAHelper:
    """
    EMA = Exponential Moving Average (moyenne glissante) des poids.

    Concept (débutant) :
    - pendant l'entraînement, les poids bougent beaucoup et peuvent être "bruités"
    - EMA garde une version "plus stable" du modèle
    - souvent cette version EMA génère de meilleures images
    """
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
            locs = inner_module.locals
            module_copy = type(inner_module)(*locs).to(self.device)
            module_copy.load_state_dict(inner_module.state_dict())
            if isinstance(module, nn.DataParallel):
                module_copy = nn.DataParallel(module_copy)
        else:
            locs = module.locals
            module_copy = type(module)(*locs).to(self.device)
            module_copy.load_state_dict(module.state_dict())

        for name, param in module_copy.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)
        return module_copy

def grad_gauss(x, m, var):
    """
    Concept :
    - gradient d'une distribution Gaussienne
    - sert ici à pousser les échantillons vers une distribution cible simple
    (utile pour initialiser / guider certaines étapes)
    """
    return -(x - m) / var

class Langevin(torch.nn.Module):
    """
    Implémente une trajectoire de sampling par dynamique de Langevin.

    Concept débutant :
    - on part d'un état initial (images ou bruit)
    - on applique plusieurs petites mises à jour
    - à chaque étape, on ajoute un bruit aléatoire
    - le réseau (net) sert à guider ces mises à jour
    """
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
        x = init_samples
        N = x.shape[0]
        steps = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        
        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)

        for k in range(self.num_steps):
            gamma = self.gammas[k]
            net_out = net(x, steps[:, k, :])
            if self.mean_match:
                t_old = net_out
            else:
                t_old = x + net_out
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma) * z
            net_out_new = net(x, steps[:, k, :])
            if self.mean_match:
                t_new = net_out_new
            else:
                t_new = x + net_out_new

            x_tot[:, k, :] = x
            out[:, k, :] = (t_old - t_new)

        return x_tot, out, steps

class CacheLoader(Dataset):
    """
    Dataset "fabriqué" à la volée.

    Concept :
    - au lieu d'entraîner directement sur les images, on génère des trajectoires
      (via Langevin + un réseau) et on les "met en cache"
    - ça transforme un problème complexe en un dataset supervisé :
      (x, out, steps) où out est la cible à prédire.
    """
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

                x = x.cpu().unsqueeze(2)
                out = out.cpu().unsqueeze(2)
                batch_data = torch.cat((x, out), dim=2)
                
                self.data[b] = batch_data.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = steps_expanded.cpu().flatten(start_dim=0, end_dim=1)
                
                del x, out, batch_data

        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], self.steps_data[index]

    def __len__(self):
        return self.data.shape[0]
