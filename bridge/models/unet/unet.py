"""
Ce fichier définit l’architecture principale UNetModel (et SuperResModel).
Le UNet est le réseau utilisé pour traiter des images dans les modèles de diffusion :
- il descend en résolution pour capturer le contexte global
- puis remonte en résolution pour reconstruire des détails
- il utilise des connexions "skip" pour ne pas perdre d’informations
Le réseau est conditionné par le temps (timesteps) et peut inclure de l’attention à certaines résolutions.
"""


from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .layers import *


class UNetModel(nn.Module):
    """
    UNet avec :
    - un encodage du temps (timesteps)
    - des blocs résiduels (ResBlock)
    - de l’attention à certaines résolutions (AttentionBlock)

    Concept UNet :
    1) on "descend" en résolution (downsampling) pour comprendre le contexte global
    2) on passe par un "milieu" (middle block)
    3) on "remonte" en résolution (upsampling) pour reconstruire des détails
    4) on utilise des "skip connections" : on garde des infos de la descente
       et on les réinjecte pendant la remontée pour ne pas perdre les détails.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        # Stocke les hyperparamètres (utile pour reproduire/charger un modèle)
        self.locals = [ in_channels,
                        model_channels,
                        out_channels,
                        num_res_blocks,
                        attention_resolutions,
                        dropout,
                        channel_mult,
                        conv_resample,
                        dims,
                        num_classes,
                        use_checkpoint,
                        num_heads,
                        num_heads_upsample,
                        use_scale_shift_norm
                    ]
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        # Encodage du temps :
        # transforme un "timestep" en un vecteur qui sera utilisé partout dans le réseau.
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        # Option : modèle conditionné par une classe (ex: générer une catégorie précise)
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # input_blocks = la partie "descente" du UNet
        # On commence par une convolution pour passer dans l’espace de features.
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        # On construit ensuite plusieurs niveaux :
        # - à chaque niveau, on applique des ResBlocks
        # - parfois, on ajoute Attention
        # - puis on downsample pour aller vers une résolution plus basse

        # middle_block = le fond du UNet
        # Résolution la plus basse : beaucoup de contexte, moins de détail spatial.
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        # output_blocks = la partie "remontée" du UNet
        # On remonte en résolution et on concatène les features de la descente (skip connections).
        
        # out = dernière partie : remet dans le nombre de canaux souhaité (ex: prédire un bruit / une image)
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )


    def convert_to_fp16(self):
        # Passe les blocs principaux en float16 (gain mémoire/perf)
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
         # Reviens en float32
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)


    def forward(self, x, timesteps, y=None):

        """
        Concept : faire passer x dans le UNet en tenant compte du temps (timesteps).

        Entrées :
        - x : batch d’images/features
        - timesteps : étape de diffusion (ou étape temporelle)
        - y : label (si modèle conditionnel)

        Sortie :
        - tenseur de même forme générale que x (selon out_channels),
          typiquement une prédiction liée au processus de diffusion (ex: bruit / score).
        """

        # Mise en forme : on enlève les dimensions inutiles pour avoir un vecteur de timesteps propre.
        timesteps = timesteps.squeeze()
         # Vérification : si le modèle est conditionnel (num_classes défini),
        # alors on DOIT fournir y, et inversement.
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # hs va stocker les sorties intermédiaires de la "descente" (downsampling).
        # Concept : ces valeurs serviront plus tard comme "raccourcis" (skip connections)
        # pour récupérer les détails lors de la remontée.
        hs = []

        # On transforme timesteps en vecteur riche (embedding du temps),
        # puis on l'adapte au réseau (time_embed).
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Si le modèle est conditionné par une classe :
        # on ajoute une information de classe à l'embedding du temps.
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        # h est le "signal" qui va traverser tout le UNet.
        h = x #.type(self.inner_dtype)

        # 1) DESCENTE du UNet :
        # on applique les blocs d'entrée (resblocks + downsample) et on mémorise chaque étape.
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        # 2) MILIEU du UNet :
        # partie la plus "profonde", à plus basse résolution (beaucoup de contexte global).
        h = self.middle_block(h, emb)

        # 3) REMONTÉE du UNet :
        # à chaque étape, on concatène avec un état sauvegardé de la descente
        # pour récupérer des informations fines (skip connections).
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        
        # Sécurité : on remet le même type numérique que l'entrée.
        h = h.type(x.dtype)

        # Dernière couche : produit la sortie dans le bon nombre de canaux.
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
         Concept : même passage que forward(), mais on récupère aussi les étapes intermédiaires.
        Utilité :
        - déboguer / visualiser ce que le réseau "apprend"
        - analyser les features à différentes résolutions
        """
         # Même logique que forward : on construit l'embedding temps (et classe si besoin)
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        # Dictionnaire de sortie :
        # - down : ce qui sort pendant la descente
        # - middle : la représentation au fond du UNet
        # - up : ce qui sort pendant la remontée
        result = dict(down=[], up=[])

        h = x#.type(self.inner_dtype)

        # DESCENTE : on stocke à la fois dans hs (pour les skip connections)
        # et dans result["down"] pour pouvoir les retourner.
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
         # MILIEU
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        # REMONTÉE : pareil que forward, mais on sauvegarde chaque étape.
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result



class SuperResModel(UNetModel):
    """
    Variante du UNet pour faire de la super-résolution.

    Concept :
    - le modèle reçoit une image (ou une estimation) en haute résolution
    - et une image low_res (basse résolution) qui sert de "guide"
    """

    def __init__(self, in_channels, *args, **kwargs):
        # On multiplie par 2 car on va concaténer x et l'image low_res agrandie :
        # donc on a 2 fois plus de canaux en entrée.
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        # On agrandit low_res pour qu'elle ait la même taille que x.
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
         # On concatène x et low_res agrandie (en canaux) pour donner au UNet
        # l'image "guide" en plus de l'image à traiter.
        x = th.cat([x, upsampled], dim=1)
         # Puis on appelle le forward du UNet standard.
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
         # Même idée que forward, mais on retourne aussi les features intermédiaires.
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
