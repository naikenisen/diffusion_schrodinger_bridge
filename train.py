"""
Script d'entraînement complet DSB (Diffusion Schrödinger Bridge) : HES -> CD30.

Idée générale (version débutant) :
- On veut apprendre à transformer des images HES en images type CD30.
- On utilise un modèle de type "diffusion / bridge" entraîné en plusieurs passes.
- Le script contient tout : chargement des données, entraînement, sauvegardes, plots.

Le modèle (UNet), le dataset (HES_CD30) et la config (cfg) sont importés depuis d'autres fichiers.
"""

import os
import sys
import copy
import time
import random
import datetime
from itertools import repeat

import numpy as np
import matplotlib
matplotlib.use('Agg') # Permet de faire des figures même sans écran (serveur)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator # Gère GPU/multi-GPU/mixed precision plus simplement

from pytorch_lightning.loggers import CSVLogger as _CSVLogger

import config as cfg
from bridge.models.unet import UNetModel
from bridge.data.hes_cd30 import HES_CD30

# Petit raccourci pour construire facilement une pipeline de transformations d'images
cmp = lambda x: transforms.Compose([*x])


# ============================================================================
#  REPEATER (boucle infinie sur un DataLoader)
# ============================================================================

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


# ============================================================================
#  LOGGER
# ============================================================================

class Logger:
    """
    Interface minimale pour enregistrer des métriques (loss, etc.).
    """
    def log_metrics(self, metric_dict, step=None, save=False):
        pass

    def log_hparams(self, hparams_dict):
        pass


class CSVLogger(Logger):
    """
    Logger qui écrit les métriques dans des fichiers CSV.

    Concept :
    - garder une trace de l'entraînement (loss, normes de gradient, etc.)
    - pouvoir relire ensuite dans Excel / Python / etc.
    """
    def __init__(self, directory='./', name='logs', save_stride=1):
        self.logger = _CSVLogger(directory, name=name)
        self.count = 0
        self.stride = save_stride

    def log_metrics(self, metrics, step=None, save=False):
        # Ajoute des métriques (ex: loss) au logger
        self.count += 1
        self.logger.log_metrics(metrics, step=step)
        # Sauvegarde périodique (évite de perdre les données en cas de crash)
        if self.count % self.stride == 0:
            self.logger.save()
            self.logger.metrics = []
        # "reset" occasionnel pour éviter de garder trop en mémoire
        if self.count > self.stride * 10:
            self.count = 0
        if save:
            self.logger.save()

    def log_hparams(self, hparams_dict):
        # Enregistre la config (hyperparamètres) dans les logs
        self.logger.log_hyperparams(hparams_dict)
        self.logger.save()


# ============================================================================
#  PLOTTER
# ============================================================================

def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    """
    Concept :
    - On a une suite d'images (png) à différentes étapes d'un processus.
    - On les assemble en GIF pour visualiser l'évolution.
    """
    frames = [Image.open(fn) for fn in plot_paths]
    frames[0].save(
        os.path.join(output_directory, f'{gif_name}.gif'),
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0,
    )


class ImPlotter:
    """
    Outil pour sauvegarder des grilles d'images (début / fin / étapes intermédiaires)
    et éventuellement créer un GIF.

    Concept :
    - visualiser la "trajectoire" de génération : comment une image évolue
      au fil des étapes de Langevin / diffusion.
    """
    def __init__(self, im_dir='./im', gif_dir='./gif', plot_level=3):
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)
        self.im_dir = im_dir
        self.gif_dir = gif_dir
        self.num_plots = 100
        self.num_digits = 20
        self.plot_level = plot_level

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        """
        initial_sample : images de départ
        x_tot_plot     : images générées au fil du temps (trajectoire)
        i              : index d'itération d'entraînement
        n              : itération IPF
        forward_or_backward : 'f' ou 'b' (sens entraîné)
        """
        if self.plot_level > 0:
             # On limite le nombre d'images affichées pour éviter des fichiers énormes
            x_tot_plot = x_tot_plot[:, :self.num_plots]
            name = '{0}_{1}_{2}'.format(forward_or_backward, n, i)
            im_dir = os.path.join(self.im_dir, name)

            if not os.path.isdir(im_dir):
                os.mkdir(im_dir)

            # Niveau 1 : sauvegarde grille début + grille fin
            if self.plot_level > 0:
                plt.clf()
                filename_grid_png = os.path.join(im_dir, 'im_grid_first.png')
                vutils.save_image(initial_sample, filename_grid_png, nrow=10)
                filename_grid_png = os.path.join(im_dir, 'im_grid_final.png')
                vutils.save_image(x_tot_plot[-1], filename_grid_png, nrow=10)

             # Niveau 2 : sauvegarde étapes intermédiaires + création GIF
            if self.plot_level >= 2:
                plt.clf()
                plot_paths = []
                num_steps, num_particles, channels, H, W = x_tot_plot.shape
                plot_steps = np.linspace(0, num_steps - 1, self.num_plots, dtype=int)

                for k in plot_steps:
                    filename_grid_png = os.path.join(im_dir, 'im_grid_{0}.png'.format(k))
                    plot_paths.append(filename_grid_png)
                    vutils.save_image(x_tot_plot[k], filename_grid_png, nrow=10)

                make_gif(plot_paths, output_directory=self.gif_dir, gif_name=name)

    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward)


# ============================================================================
#  EMA HELPER
# ============================================================================

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
        self.shadow = {} # copie des poids "lissés"
        self.device = device

    def register(self, module):
         # On mémorise une copie des poids au départ
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
          # Mise à jour EMA : nouveau = (1-mu)*poids_actuels + mu*poids_ema
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
         # Applique les poids EMA à un module (remplace les poids du module)
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
           # Crée une copie du modèle et y applique l'EMA (pratique pour échantillonner)
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
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# ============================================================================
#  LANGEVIN DYNAMICS
# ============================================================================

def grad_gauss(x, m, var):
    """
    Concept :
    - gradient d'une distribution Gaussienne
    - sert ici à pousser les échantillons vers une distribution cible simple
    (utile pour initialiser / guider certaines étapes)
    """
    return -(x - m) / var


def ornstein_ulhenbeck(x, gradx, gamma):
    """
    Concept :
    - une mise à jour type "dynamique stochastique"
    - on avance dans une direction (gradx) + on ajoute un bruit aléatoire
    """
    return x + gamma * gradx + torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)


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

         # Prépare une version "étendue" des gammas au bon format
        gammas_vec = torch.ones(self.num_steps, *self.d, device=device)
        for k in range(num_steps):
            gammas_vec[k] = gammas[k].float()
        self.gammas_vec = gammas_vec

        self.device = device if device is not None else gammas.device

        # self.time contient les temps cumulés (une échelle temporelle)
        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas, 0).to(self.device).float()

         # time_sampler : sert à tirer des pas de temps selon une distribution (pondération)
        self.time_sampler = time_sampler

    def record_init_langevin(self, init_samples):
        """
        Trajectoire spéciale utilisée au tout début (n=1 et fb='b' dans ton code).

        Concept :
        - générer une trajectoire sans réseau (ou avec une règle simple)
        - produire aussi 'out' (la "cible" à apprendre) pour l'entraînement
        """
        mean_final = self.mean_final
        var_final = self.var_final
        x = init_samples
        N = x.shape[0]

         # steps_expanded = temps associé à chaque étape pour chaque échantillon du batch
        time = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        steps_expanded = time

        # x_tot : sauvegarde les états successifs (pour plots / cache)
        # out   : sauvegarde un signal "à prédire" (utilisé comme target pendant l'entraînement)
        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        num_iter = self.num_steps

        for k in range(num_iter):
            gamma = self.gammas[k]
             # Ici : on pousse x vers une gaussienne cible (mean_final/var_final)
            gradx = grad_gauss(x, mean_final, var_final)
              # t_old / t_new : deux évaluations utilisées pour construire la cible 'out'
            t_old = x + gamma * gradx
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma) * z
            gradx = grad_gauss(x, mean_final, var_final)
            t_new = x + gamma * gradx
            x_tot[:, k, :] = x
            out[:, k, :] = (t_old - t_new)

        return x_tot, out, steps_expanded

    def record_langevin_seq(self, net, init_samples, t_batch=None, ipf_it=0, sample=False):
        """
        Trajectoire standard : le réseau 'net' guide les mises à jour.

        Concept :
        - on simule une évolution en plusieurs étapes
        - on enregistre la trajectoire x_tot (pour plots)
        - et on construit 'out' qui sert de cible pour entraîner l'autre réseau
        """
        mean_final = self.mean_final
        var_final = self.var_final
        x = init_samples
        N = x.shape[0]
        time = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        steps = time
        steps_expanded = steps

        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        num_iter = self.num_steps

        # mean_match : change la façon dont on interprète la sortie du réseau
        # (soit le réseau donne directement un point "moyen", soit une correction)
        if self.mean_match:
            for k in range(num_iter):
                gamma = self.gammas[k]

                 # t_old : prédiction du réseau à l'étape k
                t_old = net(x, steps[:, k, :])

                  # sampling : à la dernière étape, on peut choisir de ne plus ajouter de bruit
                if sample and (k == num_iter - 1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                t_new = net(x, steps[:, k, :])
                
                x_tot[:, k, :] = x
                out[:, k, :] = (t_old - t_new)
        else:
            for k in range(num_iter):
                gamma = self.gammas[k]
                t_old = x + net(x, steps[:, k, :])
                if sample and (k == num_iter - 1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                t_new = x + net(x, steps[:, k, :])
                x_tot[:, k, :] = x
                out[:, k, :] = (t_old - t_new)

        return x_tot, out, steps_expanded

    def forward(self, net, init_samples, t_batch, ipf_it):
        return self.record_langevin_seq(net, init_samples, t_batch, ipf_it)


# ============================================================================
#  CACHE LOADER
# ============================================================================

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
                 mean, std, batch_size, device='cpu',
                 dataloader_f=None, transfer=False):
        super().__init__()
        start = time.time()
        shape = langevin.d
        num_steps = langevin.num_steps

        # self.data stocke pour chaque étape :
        # - x : l'état
        # - out : la cible d'entraînement associée
        self.data = torch.zeros(
            (num_batches, batch_size * num_steps, 2, *shape)).to(device)

        # self.steps_data stocke le temps associé à chaque échantillon/étape
        self.steps_data = torch.zeros(
            (num_batches, batch_size * num_steps, 1)).to(device)

        with torch.no_grad():
            for b in range(num_batches):

                # Choix de la source des échantillons initiaux :
                # - soit on prend des images du dataset (si fb == 'b')
                # - soit on prend des images de l'autre domaine si transfer
                # - soit on part d'un bruit gaussien (sinon)
                if fb == 'b':
                    batch = next(dataloader_b)[0].to(device)
                elif fb == 'f' and transfer:
                    batch = next(dataloader_f)[0].to(device)
                else:
                    batch = mean + std * torch.randn((batch_size, *shape), device=device)

                # Première itération : trajectoire spéciale d'init
                if (n == 1) and (fb == 'b'):
                    x, out, steps_expanded = langevin.record_init_langevin(batch)
                else:
                    # Trajectoire guidée par sample_net
                    x, out, steps_expanded = langevin.record_langevin_seq(sample_net, batch, ipf_it=n)

                # On regroupe (x, out) ensemble puis on "aplatit" en une liste d'exemples
                x = x.unsqueeze(2)
                out = out.unsqueeze(2)
                batch_data = torch.cat((x, out), dim=2)
                flat_data = batch_data.flatten(start_dim=0, end_dim=1)
                self.data[b] = flat_data

                flat_steps = steps_expanded.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = flat_steps

        # Aplatit tout en un grand dataset
        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

        stop = time.time()
        print('Cache size: {0}'.format(self.data.shape))
        print("Load time: {0}".format(stop - start))

    def __getitem__(self, index):
        # Retourne : état x, cible out, et temps steps
        item = self.data[index]
        x = item[0]
        out = item[1]
        steps = self.steps_data[index]
        return x, out, steps

    def __len__(self):
        return self.data.shape[0]



# ============================================================================
#  CONFIG GETTERS (modèle, optimiseur, données, plotter, logger)
# ============================================================================

def get_models():
    """
    Construit deux réseaux UNet :
    - net_f : réseau "forward"
    - net_b : réseau "backward"

    Concept :
    - le Schrödinger Bridge entraîne deux directions (aller/retour)
    - les deux réseaux apprennent à se "répondre" via IPF.
    """
    image_size = cfg.IMAGE_SIZE
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    # On convertit les résolutions en "facteurs de downsampling"
    attention_ds = []
    for res in cfg.ATTENTION_RESOLUTIONS.split(","):
        attention_ds.append(image_size // int(res))

    # Paramètres du UNet pris depuis la config
    kwargs = {
        "in_channels": cfg.CHANNELS,
        "model_channels": cfg.NUM_CHANNELS,
        "out_channels": cfg.CHANNELS,
        "num_res_blocks": cfg.NUM_RES_BLOCKS,
        "attention_resolutions": tuple(attention_ds),
        "dropout": cfg.DROPOUT,
        "channel_mult": channel_mult,
        "num_classes": None,
        "use_checkpoint": cfg.USE_CHECKPOINT,
        "num_heads": cfg.NUM_HEADS,
        "num_heads_upsample": cfg.NUM_HEADS_UPSAMPLE,
        "use_scale_shift_norm": cfg.USE_SCALE_SHIFT_NORM,
    }
    net_f, net_b = UNetModel(**kwargs), UNetModel(**kwargs)
    return net_f, net_b


def get_optimizers(net_f, net_b, lr):
    """
    Créé les optimiseurs (ici Adam) pour chaque réseau.
    """
    return (torch.optim.Adam(net_f.parameters(), lr=lr),
            torch.optim.Adam(net_b.parameters(), lr=lr))


def get_datasets():
    """
    Charge deux datasets séparés :
    - init_ds  : images de départ (HES)
    - final_ds : images cibles (CD30)

    Concept :
    - on ne donne pas la paire (HES, CD30) directement
    - on a deux domaines séparés, et le bridge apprend à passer de l'un à l'autre.
    """
    train_transform = [
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ]
    if cfg.RANDOM_FLIP:
        train_transform.insert(2, transforms.RandomHorizontalFlip())

    root = os.path.join(cfg.DATA_DIR, 'dataset_v2')

    init_ds = HES_CD30(root, image_size=cfg.IMAGE_SIZE,
                       domain='HES', transform=cmp(train_transform))
    final_ds = HES_CD30(root, image_size=cfg.IMAGE_SIZE,
                        domain='CD30', transform=cmp(train_transform))

    # Paramètres d'une distribution simple, utilisée quand on ne part pas d'images réelles
    mean_final = torch.tensor(0.)
    var_final = torch.tensor(1. * 10 ** 3)

    return init_ds, final_ds, mean_final, var_final


def get_plotter():
    """Crée l'outil de visualisation."""
    return ImPlotter(plot_level=cfg.PLOT_LEVEL)


def get_logger(name='logs'):
    """Choisit le logger selon la config."""
    if cfg.LOGGER == 'CSV':
        return CSVLogger(directory=cfg.CSV_LOG_DIR, name=name)
    return Logger()


# ============================================================================
#  IPF BASE
# ============================================================================

# ============================================================================
#  IPF BASE
# ============================================================================

class IPFBase(torch.nn.Module):
    """
    Classe "socle" : prépare tout ce qu'il faut pour entraîner un Schrödinger Bridge.

    Concept (débutant) :
    - On entraîne 2 réseaux : forward (f) et backward (b)
    - On alterne leur entraînement via IPF (Iterative Proportional Fitting)
    - Pour entraîner un réseau, on génère d'abord des trajectoires avec l'autre réseau
      (CacheLoader), puis on fait une optimisation classique (MSE).
    """

    def __init__(self):
        super().__init__()

        # Accelerator simplifie la gestion CPU/GPU/multi-GPU (et parfois mixed precision).
        self.accelerator = Accelerator(mixed_precision="no", cpu=(cfg.DEVICE == 'cpu'))
        self.device = self.accelerator.device

        # -------------------------
        # Hyperparamètres d'entraînement
        # -------------------------
        self.n_ipf = cfg.N_IPF                 # nombre d'itérations IPF (cycles b puis f)
        self.num_steps = cfg.NUM_STEPS         # nombre d'étapes dans la trajectoire Langevin
        self.batch_size = cfg.BATCH_SIZE       # batch size utilisé pour l'optimisation
        self.num_iter = cfg.NUM_ITER           # nombre d'itérations d'optimisation par étape IPF
        self.grad_clipping = cfg.GRAD_CLIPPING # active/désactive le clipping des gradients
        self.fast_sampling = cfg.FAST_SAMPLING # option (selon config) pour accélérer le sampling
        self.lr = cfg.LR                       # learning rate

        # -------------------------
        # Construction des pas de temps (gammas)
        # -------------------------
        # Concept :
        # - on définit des petits pas gamma (taille des mises à jour)
        # - on crée une séquence symétrique (monte puis redescend) pour le bridge
        n = self.num_steps // 2
        if cfg.GAMMA_SPACE == 'linspace':
            gamma_half = np.linspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
        elif cfg.GAMMA_SPACE == 'geomspace':
            gamma_half = np.geomspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
        gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        gammas = torch.tensor(gammas).to(self.device)

        # T = "durée totale" (somme des pas), utilisée pour transformer/recentrer le temps
        self.T = torch.sum(gammas)

        # -------------------------
        # Modèles + EMA
        # -------------------------
        # Concept :
        # - build_models : crée net_f et net_b
        # - build_ema : initialise les EMA (moyennes glissantes des poids)
        self.build_models()
        self.build_ema()

        # -------------------------
        # Optimiseurs
        # -------------------------
        # 1 optimiseur par réseau (forward et backward)
        self.build_optimizers()

        # -------------------------
        # Loggers
        # -------------------------
        # logger : métriques d'entraînement (loss, grad_norm, ...)
        # save_logger : métriques liées aux samples/plots
        self.logger = get_logger()
        self.save_logger = get_logger('plot_logs')

        # -------------------------
        # Données (DataLoaders)
        # -------------------------
        # Construit les DataLoaders pour :
        # - échantillonnage / plots
        # - création de cache (trajectoires)
        self.build_dataloaders()

        # -------------------------
        # Time sampler (pondération des temps)
        # -------------------------
        # Concept :
        # - selon cfg.WEIGHT_DISTRIB, on peut donner plus de poids à certaines étapes
        #   lors du tirage (utile pour l'entraînement, selon la stratégie choisie)
        if cfg.WEIGHT_DISTRIB:
            alpha = cfg.WEIGHT_DISTRIB_ALPHA
            prob_vec = (1 + alpha) * torch.sum(gammas) - torch.cumsum(gammas, 0)
        else:
            prob_vec = gammas * 0 + 1
        time_sampler = torch.distributions.categorical.Categorical(prob_vec)

        # -------------------------
        # Définir la "forme" des données et créer l'objet Langevin
        # -------------------------
        # On récupère un batch pour connaître la taille (C,H,W)
        batch = next(self.save_init_dl)[0]
        shape = batch[0].shape
        self.shape = shape

        # Langevin = moteur qui génère des trajectoires + des cibles supervisées (out)
        self.langevin = Langevin(
            self.num_steps, shape, gammas,
            time_sampler, device=self.device,
            mean_final=self.mean_final, var_final=self.var_final,
            mean_match=cfg.MEAN_MATCH
        )

        # -------------------------
        # Gestion des checkpoints / reprise
        # -------------------------
        # Concept :
        # - possibilité de reprendre à une itération IPF donnée (checkpoint_it)
        # - et de reprendre sur la passe forward ou backward (checkpoint_pass)
        date = str(datetime.datetime.now())[0:10]
        self.name_all = date

        self.checkpoint_run = cfg.CHECKPOINT_RUN
        if cfg.CHECKPOINT_RUN:
            self.checkpoint_it = cfg.CHECKPOINT_IT
            self.checkpoint_pass = cfg.CHECKPOINT_PASS
        else:
            self.checkpoint_it = 1
            self.checkpoint_pass = 'b'

        # Outil de visualisation (grilles + gifs)
        self.plotter = get_plotter()

        # Création des dossiers de sortie (une seule fois sur le process principal)
        if self.accelerator.process_index == 0:
            os.makedirs('./im', exist_ok=True)
            os.makedirs('./gif', exist_ok=True)
            os.makedirs('./checkpoints', exist_ok=True)

        # Strides : fréquence de sauvegarde/plots et fréquence de log
        self.stride = cfg.GIF_STRIDE
        self.stride_log = cfg.LOG_STRIDE

    def build_models(self, forward_or_backward=None):
        """
        Construit les deux réseaux (forward et backward).

        forward_or_backward :
        - None : construit les deux
        - 'f'  : reconstruit uniquement le forward
        - 'b'  : reconstruit uniquement le backward
        """
        net_f, net_b = get_models()

        # Si reprise : charge des checkpoints si fournis dans la config
        if cfg.CHECKPOINT_RUN:
            if cfg.SAMPLE_CHECKPOINT_F:
                net_f.load_state_dict(torch.load(cfg.SAMPLE_CHECKPOINT_F))
            if cfg.SAMPLE_CHECKPOINT_B:
                net_b.load_state_dict(torch.load(cfg.SAMPLE_CHECKPOINT_B))

        # Option : paralléliser sur plusieurs GPUs via DataParallel
        if cfg.DATAPARALLEL:
            net_f = torch.nn.DataParallel(net_f)
            net_b = torch.nn.DataParallel(net_b)

        # Création initiale des 2 réseaux
        if forward_or_backward is None:
            net_f = net_f.to(self.device)
            net_b = net_b.to(self.device)
            self.net = torch.nn.ModuleDict({'f': net_f, 'b': net_b})

        # Remplacement uniquement du forward
        if forward_or_backward == 'f':
            net_f = net_f.to(self.device)
            self.net.update({'f': net_f})

        # Remplacement uniquement du backward
        if forward_or_backward == 'b':
            net_b = net_b.to(self.device)
            self.net.update({'b': net_b})

    def accelerate(self, forward_or_backward):
        """
        Prépare le modèle et l'optimiseur avec Accelerator.

        Concept :
        - selon le contexte, Accelerator wrap le modèle/l'optimiseur
          pour faire tourner correctement sur le(s) device(s).
        """
        (self.net[forward_or_backward], self.optimizer[forward_or_backward]) = self.accelerator.prepare(
            self.net[forward_or_backward], self.optimizer[forward_or_backward])

    def update_ema(self, forward_or_backward):
        """
        Initialise/relance l'EMA pour une direction (f ou b).

        Concept :
        - EMA garde une version "lissée" des poids
        - très utile pour faire des samples plus stables
        """
        if cfg.EMA:
            self.ema_helpers[forward_or_backward] = EMAHelper(
                mu=cfg.EMA_RATE, device=self.device)
            self.ema_helpers[forward_or_backward].register(
                self.net[forward_or_backward])

    def build_ema(self):
        """
        Crée les EMA pour forward et backward.

        En mode reprise :
        - on peut initialiser l'EMA à partir de checkpoints de "sample nets".
        """
        if cfg.EMA:
            self.ema_helpers = {}
            self.update_ema('f')
            self.update_ema('b')

            if cfg.CHECKPOINT_RUN:
                sample_net_f, sample_net_b = get_models()

                if cfg.SAMPLE_CHECKPOINT_F:
                    sample_net_f.load_state_dict(torch.load(cfg.SAMPLE_CHECKPOINT_F))
                    if cfg.DATAPARALLEL:
                        sample_net_f = torch.nn.DataParallel(sample_net_f)
                    sample_net_f = sample_net_f.to(self.device)
                    self.ema_helpers['f'].register(sample_net_f)

                if cfg.SAMPLE_CHECKPOINT_B:
                    sample_net_b.load_state_dict(torch.load(cfg.SAMPLE_CHECKPOINT_B))
                    if cfg.DATAPARALLEL:
                        sample_net_b = torch.nn.DataParallel(sample_net_b)
                    sample_net_b = sample_net_b.to(self.device)
                    self.ema_helpers['b'].register(sample_net_b)

    def build_optimizers(self):
        """
        Crée les optimiseurs (ici Adam) pour les deux réseaux.
        """
        optimizer_f, optimizer_b = get_optimizers(self.net['f'], self.net['b'], self.lr)
        self.optimizer = {'f': optimizer_f, 'b': optimizer_b}

    def build_dataloaders(self):
        """
        Prépare les DataLoaders.

        Concept :
        - save_* : petits batches pour visualiser/plotter
        - cache_* : batches utilisés pour construire le dataset de cache (trajectoires)
        - repeater(...) : permet d'appeler next(...) sans fin (pas de gestion d'epoch)
        """
        init_ds, final_ds, mean_final, var_final = get_datasets()

        # Paramètres utilisés si on génère depuis une distribution simple (bruit)
        self.mean_final = mean_final.to(self.device)
        self.var_final = var_final.to(self.device)
        self.std_final = torch.sqrt(var_final).to(self.device)

        # Rend les workers reproductibles (mais différents entre eux)
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id + self.accelerator.process_index)

        self.kwargs = {
            "num_workers": cfg.NUM_WORKERS,
            "pin_memory": cfg.PIN_MEMORY,
            "worker_init_fn": worker_init_fn,
            "drop_last": True
        }

        # Domaine initial (HES)
        self.save_init_dl = DataLoader(init_ds, batch_size=cfg.PLOT_NPAR, shuffle=True, **self.kwargs)
        self.cache_init_dl = DataLoader(init_ds, batch_size=cfg.CACHE_NPAR, shuffle=True, **self.kwargs)

        # Accelerator prépare les dataloaders (utile en multi-GPU)
        (self.cache_init_dl, self.save_init_dl) = self.accelerator.prepare(self.cache_init_dl, self.save_init_dl)

        # Itérateurs infinis
        self.cache_init_dl = repeater(self.cache_init_dl)
        self.save_init_dl = repeater(self.save_init_dl)

        # Si TRANSFER : on utilise aussi le domaine final (CD30) comme source réelle
        if cfg.TRANSFER:
            self.save_final_dl = DataLoader(final_ds, batch_size=cfg.PLOT_NPAR, shuffle=True, **self.kwargs)
            self.cache_final_dl = DataLoader(final_ds, batch_size=cfg.CACHE_NPAR, shuffle=True, **self.kwargs)

            (self.cache_final_dl, self.save_final_dl) = self.accelerator.prepare(self.cache_final_dl, self.save_final_dl)

            self.cache_final_dl = repeater(self.cache_final_dl)
            self.save_final_dl = repeater(self.save_final_dl)
        else:
            self.cache_final_dl = None
            self.save_final_dl = None

    def new_cacheloader(self, forward_or_backward, n, use_ema=True):
        """
        Crée un DataLoader "cache" pour entraîner une direction.

        Concept :
        - Pour entraîner 'b', on génère des trajectoires avec 'f'
        - Pour entraîner 'f', on génère des trajectoires avec 'b'
        - On stocke ces trajectoires sous forme (x, out, steps) dans CacheLoader
        """
        # Direction utilisée pour générer les trajectoires (l'autre réseau)
        sample_direction = 'f' if forward_or_backward == 'b' else 'b'

        # On préfère souvent le réseau EMA pour sampler (plus stable)
        if use_ema:
            sample_net = self.ema_helpers[sample_direction].ema_copy(self.net[sample_direction])
        else:
            sample_net = self.net[sample_direction]

        # Construction du dataset de cache selon la direction entraînée
        if forward_or_backward == 'b':
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader(
                'b', sample_net, self.cache_init_dl,
                cfg.NUM_CACHE_BATCHES, self.langevin, n,
                mean=None, std=None,
                batch_size=cfg.CACHE_NPAR,
                device=self.device,
                dataloader_f=self.cache_final_dl,
                transfer=cfg.TRANSFER
            )
        else:
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader(
                'f', sample_net, None,
                cfg.NUM_CACHE_BATCHES, self.langevin, n,
                mean=self.mean_final, std=self.std_final,
                batch_size=cfg.CACHE_NPAR,
                device=self.device,
                dataloader_f=self.cache_final_dl,
                transfer=cfg.TRANSFER
            )

        # DataLoader + préparation Accelerator + itérateur infini
        new_dl = DataLoader(new_dl, batch_size=self.batch_size)
        new_dl = self.accelerator.prepare(new_dl)
        new_dl = repeater(new_dl)
        return new_dl

    def train(self):
        # À implémenter dans la classe enfant (IPFSequential)
        pass

    def save_step(self, i, n, fb):
        """
        Sauvegarde périodique + génération de samples pour suivi visuel.

        Concept :
        - à certains pas (stride), on :
          1) sauvegarde les poids
          2) génère une trajectoire (sampling)
          3) sauvegarde des images (grid) + gifs + logs simples
        """
        if self.accelerator.is_local_main_process:
            if ((i % self.stride == 0) or (i % self.stride == 1)) and (i > 0):

                # Choix du modèle de sampling (EMA ou non)
                if cfg.EMA:
                    sample_net = self.ema_helpers[fb].ema_copy(self.net[fb])
                else:
                    sample_net = self.net[fb]

                # -------------------------
                # 1) Sauvegarde du réseau courant
                # -------------------------
                name_net = 'net_' + fb + '_' + str(n) + "_" + str(i) + '.ckpt'
                name_net_ckpt = './checkpoints/' + name_net

                if cfg.DATAPARALLEL:
                    torch.save(self.net[fb].module.state_dict(), name_net_ckpt)
                else:
                    torch.save(self.net[fb].state_dict(), name_net_ckpt)

                # -------------------------
                # 2) Sauvegarde du réseau EMA (si activé)
                # -------------------------
                if cfg.EMA:
                    name_net = 'sample_net_' + fb + '_' + str(n) + "_" + str(i) + '.ckpt'
                    name_net_ckpt = './checkpoints/' + name_net
                    if cfg.DATAPARALLEL:
                        torch.save(sample_net.module.state_dict(), name_net_ckpt)
                    else:
                        torch.save(sample_net.state_dict(), name_net_ckpt)

                # -------------------------
                # 3) Sampling + plots (sans gradient)
                # -------------------------
                with torch.no_grad():
                    # Seed fixe pour avoir des images comparables d'une sauvegarde à l'autre
                    self.set_seed(seed=0 + self.accelerator.process_index)

                    # Choix des images de départ pour visualiser
                    if fb == 'f':
                        # Pour forward : on part d'images HES
                        batch = next(self.save_init_dl)[0].to(self.device)
                    elif cfg.TRANSFER:
                        # Sinon si TRANSFER : on part d'images CD30 réelles
                        batch = next(self.save_final_dl)[0].to(self.device)
                    else:
                        # Sinon : on part d'un bruit gaussien
                        batch = self.mean_final + self.std_final * torch.randn(
                            (cfg.PLOT_NPAR, *self.shape), device=self.device
                        )

                    # Génère une trajectoire complète (sample=True = dernière étape sans bruit)
                    x_tot, out, steps_expanded = self.langevin.record_langevin_seq(
                        sample_net, batch, ipf_it=n, sample=True
                    )

                    # Réorganisation des dimensions pour faire des grilles / gifs
                    shape_len = len(x_tot.shape)
                    x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
                    x_tot_plot = x_tot.detach()

                # -------------------------
                # 4) Stats simples pour surveiller la "santé" des sorties
                # -------------------------
                init_x = batch.detach().cpu().numpy()
                final_x = x_tot_plot[-1].detach().cpu().numpy()
                std_final = np.std(final_x)
                std_init = np.std(init_x)
                mean_final = np.mean(final_x)
                mean_init = np.mean(init_x)

                print('Initial variance: ' + str(std_init ** 2))
                print('Final variance: ' + str(std_final ** 2))

                # Log des stats de sampling
                self.save_logger.log_metrics({
                    'FB': fb,
                    'init_var': std_init ** 2, 'final_var': std_final ** 2,
                    'mean_init': mean_init, 'mean_final': mean_final,
                    'T': self.T,
                })

                # Sauvegarde images/gif
                self.plotter(batch, x_tot_plot, i, n, fb)

    def set_seed(self, seed=0):
        """
        Fixe les seeds pour reproductibilité.
        """
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear(self):
        """
        Libère la mémoire GPU (utile après cache/sampling).
        """
        torch.cuda.empty_cache()


# ============================================================================
#  IPF SEQUENTIAL (entraînement)
# ============================================================================

class IPFSequential(IPFBase):
    """
    Implémentation "séquentielle" de l'entraînement IPF.

    Concept (débutant) :
    - À chaque itération IPF n :
      1) on entraîne le réseau backward (b)
      2) puis on entraîne le réseau forward (f)
    - Chaque entraînement utilise un "cache" généré par l'autre réseau.
    """

    def __init__(self):
        super().__init__()

    def ipf_step(self, forward_or_backward, n):
        """
        Entraîne UNE direction ('f' ou 'b') pendant num_iter itérations, à l'itération IPF n.

        Étapes conceptuelles :
        1) Construire un dataset supervisé (CacheLoader) en générant des trajectoires
           avec le réseau opposé (direction inverse).
        2) Entraîner le réseau courant sur (x -> out) avec une loss MSE.
        3) Sauvegarder/plotter régulièrement + rafraîchir le cache de temps en temps.
        """
        # 1) Génère le cache : un DataLoader de tuples (x, out, steps)
        new_dl = self.new_cacheloader(forward_or_backward, n, cfg.EMA)

        # Option : ne pas réutiliser le réseau précédent
        # Concept :
        # - si USE_PREV_NET=False, on reconstruit le réseau avant de l'entraîner
        # - utile si on veut repartir "proprement" à chaque itération IPF
        if not cfg.USE_PREV_NET:
            self.build_models(forward_or_backward)
            self.update_ema(forward_or_backward)

        # (Re)crée l'optimiseur et prépare modèle+optimiseur avec Accelerator
        self.build_optimizers()
        self.accelerate(forward_or_backward)

        # 2) Boucle d'optimisation classique
        for i in tqdm(range(self.num_iter + 1)):
            # Seed dépendant de (n, i) : reproductibilité + diversité entre itérations
            self.set_seed(seed=n * self.num_iter + i)

            # Récupère un batch du cache :
            # - x : état / image intermédiaire
            # - out : cible supervisée à prédire
            # - steps_expanded : temps associé
            x, out, steps_expanded = next(new_dl)
            x = x.to(self.device)
            out = out.to(self.device)
            steps_expanded = steps_expanded.to(self.device)

            # eval_steps : conversion du temps (ici on utilise T - t)
            # Concept :
            # - selon la formulation du bridge, la direction peut utiliser un temps "renversé"
            eval_steps = self.T - steps_expanded

            # Prédiction du réseau :
            # - mode MEAN_MATCH : la sortie du réseau correspond à un "point moyen",
            #   donc on retire x pour obtenir une correction comparable à out.
            # - sinon : la sortie du réseau est directement la correction.
            if cfg.MEAN_MATCH:
                pred = self.net[forward_or_backward](x, eval_steps) - x
            else:
                pred = self.net[forward_or_backward](x, eval_steps)

            # Loss supervisée : on veut que pred ≈ out
            loss = F.mse_loss(pred, out)

            # Backprop (Accelerator gère correctement le backward selon le contexte)
            self.accelerator.backward(loss)

            # Option : clipping des gradients pour éviter des gradients trop grands
            if self.grad_clipping:
                clipping_param = cfg.GRAD_CLIP
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net[forward_or_backward].parameters(), clipping_param
                )
            else:
                total_norm = 0.

            # Logs périodiques (loss + norme de gradient)
            if (i % self.stride_log == 0) and (i > 0):
                self.logger.log_metrics({
                    'forward_or_backward': forward_or_backward,
                    'loss': loss,
                    'grad_norm': total_norm,
                }, step=i + self.num_iter * n)

            # Step d'optimisation
            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad()

            # Mise à jour EMA (si activé) : version lissée des poids
            if cfg.EMA:
                self.ema_helpers[forward_or_backward].update(self.net[forward_or_backward])

            # Sauvegarde + sampling + plots (selon stride)
            self.save_step(i, n, forward_or_backward)

            # 3) Rafraîchissement du cache
            # Concept :
            # - comme le réseau opposé évolue, les trajectoires "idéales" changent aussi
            # - on reconstruit donc périodiquement un nouveau cache pour rester cohérent
            if (i % cfg.CACHE_REFRESH_STRIDE == 0) and (i > 0):
                new_dl = None
                torch.cuda.empty_cache()
                new_dl = self.new_cacheloader(forward_or_backward, n, cfg.EMA)

        # Nettoyage
        new_dl = None
        self.clear()

    def train(self):
        """
        Boucle principale d'entraînement IPF.

        Concept :
        - On fait d'abord une trajectoire d'initialisation (pour visualiser le point de départ).
        - Ensuite, pour chaque itération IPF n :
          - on entraîne 'b' puis 'f' (sauf cas spécial de reprise de checkpoint).
        """

        # -------------------------
        # INITIAL FORWARD PASS (visualisation)
        # -------------------------
        # On génère une trajectoire d'init (sans réseau) uniquement pour voir à quoi ressemble
        # le processus au tout début (utile pour vérifier que tout marche).
        if self.accelerator.is_local_main_process:
            init_sample = next(self.save_init_dl)[0].to(self.device)

            x_tot, _, _ = self.langevin.record_init_langevin(init_sample)

            # Mise en forme pour plot : (steps, batch, C, H, W)
            shape_len = len(x_tot.shape)
            x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
            x_tot_plot = x_tot.detach()

            # Sauvegarde grilles + éventuellement gif
            self.plotter(init_sample, x_tot_plot, 0, 0, 'f')

            # Libération mémoire
            x_tot_plot = None
            x_tot = None
            torch.cuda.empty_cache()

        # -------------------------
        # Itérations IPF
        # -------------------------
        for n in range(self.checkpoint_it, self.n_ipf + 1):
            print('IPF iteration: ' + str(n) + '/' + str(self.n_ipf))

            # Reprise : si on doit démarrer sur forward à l'itération de checkpoint
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_step('f', n)
            else:
                # En routine : backward puis forward
                self.ipf_step('b', n)
                self.ipf_step('f', n)



# ============================================================================
#  MAIN
# ============================================================================

if __name__ == '__main__':
    """
    Point d'entrée du script.

    Concept :
    - Ce bloc ne s'exécute que si on lance le fichier directement :
      python train_dsb.py
    - Il sert à afficher la configuration du run (pour vérifier qu'on n'a pas
      oublié un paramètre), puis à lancer l'entraînement.
    """

    # Affiche un résumé des paramètres importants du run
    print('=== DSB Training: HES -> CD30 ===')
    print(f'Image size : {cfg.IMAGE_SIZE}')   # taille des images (ex: 256x256)
    print(f'Batch size : {cfg.BATCH_SIZE}')   # batch size d'entraînement
    print(f'Num iter   : {cfg.NUM_ITER}')     # itérations d'optimisation par étape IPF
    print(f'Num IPF    : {cfg.N_IPF}')        # nombre d'itérations IPF (cycles b puis f)
    print(f'Num steps  : {cfg.NUM_STEPS}')    # nombre d'étapes Langevin / diffusion
    print(f'Device     : {cfg.DEVICE}')       # cpu / cuda
    print(f'Transfer   : {cfg.TRANSFER}')     # utilise (ou non) des images du domaine final
    print(f'Data dir   : {cfg.DATA_DIR}')     # chemin vers les données
    print('Directory  : ' + os.getcwd())      # dossier courant (où seront écrits logs/ckpt)

    # Crée l'objet d'entraînement IPF (prépare modèles, données, langevin, etc.)
    ipf = IPFSequential()

    # Lance la boucle principale d'entraînement (IPF)
    ipf.train()
