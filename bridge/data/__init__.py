"""
Ce fichier regroupe des fonctions utilitaires liées au pré-traitement des données.
Il contient des transformations appliquées aux images avant de les envoyer au modèle
(ex : ajout de bruit léger, changement d’échelle des valeurs, transformation logit).
L’objectif est de mettre les données dans un format plus adapté et plus stable pour l’apprentissage.
"""


import torch


def logit_transform(image: torch.Tensor, lam=1e-6):
    """" La fonction logit_transform sert à changer la "forme" des valeurs de l’image.
    Au départ, les pixels sont entre 0 et 1.
    Ici, on les transforme pour qu’ils puissent prendre n’importe quelle valeur
    (positives ou négatives), ce qui est souvent plus pratique pour entraîner un modèle.
    
    lam sert à éviter les valeurs exactement égales à 0 ou 1,
    car elles posent des problèmes mathématiques. 
    """
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)



    
def data_transform(d_config, X):
    """ La fonction data_transform prépare les images AVANT de les donner au modèle.
    Elle applique différentes transformations selon la configuration choisie.
    L’idée générale est :
        - rendre les images plus "continues"= éviter des valeurs trop "rigides" dans les pixels (0,1,2…)
                 → aide le modèle à apprendre en douceur
        - adapter les nombres pour faciliter l’apprentissage du modèle
        """
    if d_config.uniform_dequantization:
            # Ajoute un petit bruit aléatoire uniforme aux pixels.
            # Cela permet d’éviter que les valeurs soient trop "discrètes"
            # (ex: uniquement 0, 1, 2, 3...).
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    elif d_config.gaussian_dequantization:
            # Ajoute un très léger bruit gaussien aux pixels.
            # Le but est le même : lisser les valeurs pour aider l’apprentissage.
        X = X + torch.randn_like(X) * 0.01

    if d_config.rescaled:
            # Change l’échelle des pixels :
            # au lieu d’être entre 0 et 1, ils seront entre -1 et 1.
            # Beaucoup de réseaux de neurones fonctionnent mieux avec cette échelle.
        X = 2 * X - 1.
    elif d_config.logit_transform:
            # Applique la transformation logit définie plus haut,
            # pour enlever les bornes [0,1].
        X = logit_transform(X)

    return X




def inverse_data_transform(d_config, X):
    # La fonction inverse_data_transform fait l’inverse de data_transform.
    # Elle sert à remettre la sortie du modèle sous forme d’image "normale"
    # que l’on peut afficher ou sauvegarder.
    if d_config.logit_transform:
        # Ramène les valeurs vers l’intervalle [0,1]
        # après une transformation logit.
        X = torch.sigmoid(X)
    elif d_config.rescaled:
        # Ramène les valeurs de [-1,1] à [0,1].
        X = (X + 1.) / 2.

    # Sécurité : on force toutes les valeurs à rester entre 0 et 1
    # pour éviter des pixels invalides.
    return torch.clamp(X, 0.0, 1.0)