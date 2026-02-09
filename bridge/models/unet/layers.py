
import math
from abc import abstractmethod
import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
# Activation : version compatible si PyTorch n'a pas SiLU.
# Concept : fonction d’activation "douce", souvent utilisée dans les diffusions.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

# Normalisation : stabilise l’apprentissage.
# Ici, on force les calculs internes en float32 pour éviter des erreurs numériques.
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Fabrique une couche de convolution 1D / 2D / 3D selon le type de données.
    Concept :
    - même code pour images (2D), volumes (3D), signaux (1D)
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Crée une couche fully-connected (linéaire).
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Pooling moyen 1D / 2D / 3D.
    Concept :
    - réduire la taille (downsampling) en gardant une moyenne
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Exponential Moving Average (EMA) des poids.
    Concept :
    - garder une version "lissée" des poids (souvent meilleure en génération)
    - rate proche de 1 => mise à jour lente (très lissée)
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module, active=True):
    """
    Met les poids du module à 0.
    Concept :
    - démarrer certains blocs comme "neutres" au début de l’entraînement
      (pour stabiliser / contrôler l’impact d’un bloc)
    """
    if active:
        for p in module.parameters():
            p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Multiplie les poids par une constante.
    Concept :
    - ajuster l’intensité d’un bloc (utile pour init / stabilité)
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Moyenne sur toutes les dimensions sauf le batch.
    Concept :
    - calculer une moyenne "par exemple" (par image, par élément du batch)
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Normalisation standard.
    Concept :
    - stabiliser l’apprentissage (éviter explosions/instabilités)
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Encode le temps t en vecteur (sin/cos).
    Concept :
    - donner au réseau une représentation riche de l’étape de diffusion
      (comme les positional encodings des Transformers)
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Gradient checkpointing.
    Concept :
    - économiser de la mémoire pendant l’entraînement
    - en échange : recalculer certaines choses au backward (plus lent)
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    # Mécanisme interne pour faire du checkpointing (économie mémoire).
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    # Recalcule le forward pour pouvoir obtenir les gradients sans stocker
    # toutes les activations en mémoire.
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class TimestepBlock(nn.Module):
    """
    Interface : bloc qui a besoin du temps (embedding) en plus de x.
    Concept :
    - certains blocs du UNet doivent être "conditionnés" par l’étape t.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Variante de nn.Sequential qui sait passer 'emb' aux couches qui en ont besoin.
    Concept :
    - enchaîner des couches, mais garder la possibilité d'injecter l'information temps
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Agrandit une représentation (upsampling).
    Concept :
    - dans UNet, on remonte en résolution pour reconstruire une image détaillée
    - option : ajouter une convolution après l’agrandissement
    """
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Réduit la résolution (downsampling).
    Concept :
    - dans UNet, on descend en résolution pour capter du contexte global
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

"""
⚠️ Note : dans Downsample, la ligne self.op = avg_pool_nd(stride) semble bizarre : 
normalement avg_pool_nd attend dims en premier. C’est peut-être un bug/copie.
"""

class ResBlock(TimestepBlock):
    """ 
    Bloc "résiduel" (ResNet-style), utilisé partout dans le UNet.

    Idée pour débutant :
    - on applique quelques transformations à l'entrée x
    - MAIS on ajoute aussi x à la fin ("raccourci", skip connection)
    → ça aide le réseau à apprendre sans se perdre (plus stable, plus profond)

    Particularité ici :
    - le bloc est "conditionné" par le temps (emb),
      donc son comportement dépend de l'étape timesteps.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()

          # Informations de base (surtout utile pour comprendre/debug)
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout

         # Le bloc peut garder le même nombre de canaux ou en changer
        self.out_channels = out_channels or channels

        # Options d'architecture
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # Partie 1 : traitement principal de x (normaliser + activer + convolution)
        # Concept :
        # - normalisation : stabilise l'entraînement
        # - activation : rend le modèle non-linéaire
        # - convolution : extrait / transforme des motifs (features)
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # Partie 2 : transformation de l'embedding temps (emb)
        # Concept :
        # - on convertit le "temps" en informations utilisables pour moduler le bloc
        # - si use_scale_shift_norm=True, on produit 2 infos (scale et shift)
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # Partie 3 : sortie du bloc (après injection de l'information temps)
        # Concept :
        # - on affine les features
        # - dropout : limite l'overfitting
        # - zero_module(...) : initialise la dernière conv à 0 pour démarrer doucement
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # Skip connection (raccourci) :
        # Concept :
        # - si le nombre de canaux ne change pas : on peut ajouter x directement
        # - sinon : on transforme x pour qu'il ait la même forme que h avant de les additionner
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
             # Option : utiliser une convolution "classique" (3x3) pour adapter les canaux
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
             # Option : convolution 1x1 (plus simple) pour adapter uniquement les canaux
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Applique le bloc à x en tenant compte de emb (le temps).
        Concept :
        - si use_checkpoint=True : on économise de la mémoire à l'entraînement
          (mais c'est un peu plus lent)
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
         # Transforme x (chemin principal)
        h = self.in_layers(x)
        
         # Transforme emb pour influencer le bloc
        emb_out = self.emb_layers(emb).type(h.dtype)

        # On adapte la forme de emb_out pour pouvoir l'appliquer sur une image/feature map
        # (ex: passer de [N, C] à [N, C, 1, 1])
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
             # Mode "scale/shift" :
            # Concept : emb ne s'ajoute pas juste, il "modifie" la normalisation
            # (souvent plus puissant)
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
              # Mode simple :
            # Concept : on injecte le temps en l'ajoutant aux features
            h = h + emb_out
            h = self.out_layers(h)
         # Résiduel : on ajoute le raccourci (skip connection)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    Bloc d'attention.
    Idée pour débutant :
    - une convolution "voit" surtout localement (autour d'un pixel).
    - l'attention permet à une zone de l'image de "regarder" toutes les autres zones.
    → utile pour capturer des relations à longue distance (motifs éloignés).

    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        # Normalisation avant l'attention (stabilité)
        self.norm = normalization(channels)
         # On crée en une seule couche :
        # Q = Query, K = Key, V = Value (les 3 ingrédients de l'attention)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        # Calcul de l'attention
        self.attention = QKVAttention()
         # Projection finale, initialisée à 0 pour démarrer doucement
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
          # Même idée que plus haut : checkpoint optionnel pour économiser de la mémoire
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        # b = batch size, c = channels, spatial = dimensions spatiales (H,W) ou autre
        b, c, *spatial = x.shape
         # On "aplatit" l'image pour la voir comme une suite de positions
        # (T = nombre de positions = H*W)
        x = x.reshape(b, c, -1)
        # Prépare Q, K, V
        qkv = self.qkv(self.norm(x))
        # Sépare en plusieurs têtes (multi-head attention)
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
          # Calcule l'attention : mélange les infos entre positions
        h = self.attention(qkv)
         # Remet la forme batch normale
        h = h.reshape(b, -1, h.shape[-1])
         # Projection de sortie
        h = self.proj_out(h)
        # Skip connection : on ajoute l'entrée à la sortie (stabilité)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    Moteur de calcul de l'attention (à partir de Q, K, V).
    Idée pour débutant :
    - Q (query) : "qu'est-ce que je cherche ?"
    - K (key)   : "qu'est-ce que je représente ?"
    - V (value) : "quelle info je donne ?"
    L'attention calcule quelles positions doivent influencer les autres.
    """

    def forward(self, qkv):
        """
        Entrée :
        - qkv contient Q, K, V concaténés.
        Sortie :
        - une nouvelle représentation où chaque position est un mélange
          d’informations venant d’autres positions.
        """
        # On coupe qkv en 3 morceaux : Q, K, 
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
         # Normalisation pour stabiliser les produits (évite des valeurs trop grandes)
        scale = 1 / math.sqrt(math.sqrt(ch))
         # Calcule la "similarité" entre toutes les positions (poids d'attention)
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  
        # Transforme ces similarités en probabilités (somme = 1)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        # Combine les valeurs V selon ces poids
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        Sert uniquement à estimer le coût de calcul (nombre d'opérations).
        Utile pour du profiling (mesurer la lourdeur du modèle).
        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        #L'attention est coûteuse car elle compare toutes les positions entre elles.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])
