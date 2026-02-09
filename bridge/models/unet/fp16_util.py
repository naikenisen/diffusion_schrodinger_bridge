"""
Ce fichier contient des outils pour entraîner le modèle avec moins de mémoire grâce au float16 (FP16).
Le FP16 accélère souvent l’entraînement sur GPU et réduit l’utilisation mémoire, mais peut être moins stable.
Le fichier propose donc des fonctions pour :
- convertir certains modules en float16 / float32
- gérer des paramètres "maîtres" en float32 pour garder une optimisation stable.
"""

import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def convert_module_to_f16(l):
    """ 
    Convertit certains modules (convolutions) en float16.
    Concept :
        - réduire la mémoire utilisée
        - accélérer l'entraînement (selon le GPU) 
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Fait l'inverse : remet les convolutions en float32.
    Concept :
    - revenir en précision normale si besoin (stabilité, export, etc.)
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        l.bias.data = l.bias.data.float()


def make_master_params(model_params):
    """
    Crée une copie des paramètres du modèle en float32 (paramètres "maîtres").
    Concept :
    - le modèle peut calculer en float16 (rapide)
    - mais on garde une version float32 pour faire les mises à jour (plus stable)
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def model_grads_to_master_grads(model_params, master_params):
    """
    Copie les gradients calculés sur le modèle vers les paramètres maîtres.
    Concept :
    - le backward calcule des gradients sur les params du modèle
    - on récupère ces gradients en float32 pour une mise à jour plus stable
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copie les paramètres maîtres (float32) vers les paramètres du modèle.
    Concept :
    - après l'optimisation (mise à jour), on remet les nouvelles valeurs
      dans le modèle qui sert à faire les forward/backward
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    model_params = list(model_params)

    for param, master_param in zip(
        model_params, unflatten_master_params(model_params, master_params)
    ):
        param.detach().copy_(master_param)


def unflatten_master_params(model_params, master_params):
    """
    Re-transforme le gros vecteur de paramètres maîtres en une liste de tensors
    qui ont les mêmes formes que les paramètres du modèle.
    Concept :
    - "flatten" = tout mettre bout à bout
    - "unflatten" = remettre dans les formes d’origine
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def zero_grad(model_params):
    """
    Remet les gradients à zéro.
    Concept :
    - en deep learning, on veut éviter d'accumuler les gradients
      d'une itération à l'autre (sauf cas particulier).
    """
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()