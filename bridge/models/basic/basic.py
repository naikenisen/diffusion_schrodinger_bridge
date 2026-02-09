"""
Ce fichier contient une version simple de réseau de score (ScoreNetwork).
Il s’agit d’un petit réseau de type MLP qui prend en entrée :
- une donnée x (ex : position / vecteur / état)
- un temps t (timestep)
et produit une sortie correspondant à une information "de score" utilisée dans les méthodes
de diffusion / Schrödinger Bridge. C’est une version légère, plutôt utilisée sur des données simples
(par ex en 2D), pas sur des images.
"""


import torch
from .layers import MLP
from .time_embedding import get_timestep_embedding

class ScoreNetwork(torch.nn.Module):
    """Réseau de neurones utilisé pour prédire une "direction de correction" (un score)
    à partir :
        - d'un état x (ex: un point, une image, un vecteur de données)
        - d'un temps t (souvent un indice d'étape dans un processus progressif)
    
    Idée simple : le réseau apprend "comment ajuster x" en fonction de t. """

    def __init__(self, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128], x_dim=2):
        super().__init__()
        # Taille de la représentation du temps (combien de nombres servent à représenter t)
        self.temb_dim = pos_dim
        # Après l'encodage du temps, on obtient souvent 2*pos_dim
        # (car on utilise une représentation sinus/cosinus).
        t_enc_dim = pos_dim *2
        # Stocke les paramètres principaux (pratique pour debug / sauvegarde)
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]

        # Partie principale du réseau :
        # prend une représentation de x et une représentation de t,
        # les combine, puis produit une sortie de même dimension que x.
        self.net = MLP(2 * t_enc_dim,        # entrée = concat(x_enc, t_enc)
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())
        
        # Petit réseau qui transforme la représentation brute du temps
        # en représentation plus riche (compréhensible par le modèle).
        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())
        
        # Petit réseau qui transforme x (les données) en représentation plus riche
        # avant de la mélanger avec le temps.
        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

    def forward(self, x, t):
         # Assure que x a bien une dimension "batch".
        # Exemple : si x est un seul point (forme [x_dim]),
        # on le transforme en [1, x_dim] pour que le réseau fonctionne pareil.
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # Convertit le temps t en un vecteur de nombres (embedding du temps).
        # But : donner au réseau une façon "riche" de comprendre la notion de temps/étape.
        temb = get_timestep_embedding(t, self.temb_dim)
        # Rend cet embedding temps encore plus adapté au réseau (encodage appris).
        temb = self.t_encoder(temb)
        # Encode aussi x dans un espace de représentation.
        xemb = self.x_encoder(x)
        # Combine l'information "donnée" (x) et "étape" (t) dans un seul vecteur.
        h = torch.cat([xemb ,temb], -1)
        # Produit final : une sortie de dimension x_dim.
        # Dans ce type de modèle, c'est souvent une "correction" ou un "score" appliqué à x.
        out = self.net(h) 
        return out
