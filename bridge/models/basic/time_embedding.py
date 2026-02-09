"""
Ce fichier fournit une fonction d’encodage du temps (timestep embedding).
L’idée est de transformer un nombre t (étape de diffusion) en un vecteur de taille fixe,
avec des sinusoïdes (sin/cos), afin que le réseau puisse mieux utiliser l’information temporelle.
C’est le même principe que les positional encodings utilisés dans les Transformers.
"""

import torch
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, embedding_dim=128):
    """
    Cette fonction transforme un "temps" (ou numéro d’étape) en un vecteur.
    Exemple : t=10 devient une liste de nombres de taille embedding_dim.

    Pourquoi faire ça ?
    - Un réseau de neurones apprend mieux quand t n'est pas juste un nombre brut.
    - On crée une représentation plus expressive avec des sinusoïdes (sin/cos),
      comme dans les Transformers ("positional encoding").

    Résultat :
    - on obtient un vecteur qui change régulièrement avec t
    - des temps proches donnent des vecteurs "proches"
   
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    # On construit moitié sin, moitié cos (donc 2 moitiés = embedding_dim)
    half_dim = embedding_dim // 2
    # Crée une série d'échelles (fréquences) pour couvrir plusieurs "rythmes"
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)
    # Applique ces fréquences au temps
    emb = timesteps.float() * emb.unsqueeze(0)
    # Construit l'encodage final : [sin(...), cos(...)]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    # Si embedding_dim est impair, on rajoute un 0 pour compléter la taille
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb
