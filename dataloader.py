import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


def logit_transform(image: torch.Tensor, lam=1e-6):
    """
    Transformation d’espace : passer d’une représentation bornée [0,1]
    vers un espace non borné (ℝ) pour faciliter l’apprentissage numérique.
    lam évite les valeurs extrêmes (0 ou 1) qui posent problème au logit.
    """
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(d_config, X):
    """
    Pré-traitement des données d’entrée pour le modèle :
       - déquantification (réduire les artefacts liés au codage discret des pixels)
       - changement d’échelle / changement d’espace (adapter le support des données
        à ce qu’attend l’architecture ou l’objectif d’entraînement)"""
    if d_config.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    elif d_config.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if d_config.rescaled:
        X = 2 * X - 1.
    elif d_config.logit_transform:
        X = logit_transform(X)

    return X


def inverse_data_transform(d_config, X):
    """
    Post-traitement : reconvertir la sortie du modèle vers un format image interprétable.
    (retour vers [0,1] + sécurisation des bornes)
    """
    if d_config.logit_transform:
        X = torch.sigmoid(X)
    elif d_config.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)


class dataloader(Dataset):
    """
    Dataset for paired HES / CD30 virtual staining.

    Concept : fournir un accès standardisé aux images d’un domaine (HES ou CD30)
    dans un format compatible PyTorch, pour alimenter un pipeline d’entraînement.

    NB : La doc décrit des paires HES↔CD30 (même nom de fichier dans deux dossiers),
    mais cette classe, telle qu’écrite, charge un seul domaine à la fois (domain='HES' ou 'CD30').
    """
    def __init__(self, root, image_size=256, transform=None, split='train'):
        super().__init__()
        self.root = root
        self.image_size = image_size
        hes_dir = os.path.join(root, split, 'HES')
        cd30_dir = os.path.join(root, split, 'CD30')

        if not os.path.isdir(hes_dir) or not os.path.isdir(cd30_dir):
            raise RuntimeError(
                f"Dossiers {hes_dir} ou {cd30_dir} introuvables. Vérifiez dataset_v4/{split}/HES et dataset_v4/{split}/CD30."
            )

        hes_files = set(os.listdir(hes_dir))
        cd30_files = set(os.listdir(cd30_dir))
        paired_files = sorted(list(hes_files & cd30_files))
        if len(paired_files) == 0:
            raise RuntimeError(
                f"Aucune paire trouvée entre {hes_dir} et {cd30_dir}. "
                f"Vérifiez que les noms de fichiers correspondent."
            )
        self.paired_files = paired_files
        self.hes_dir = hes_dir
        self.cd30_dir = cd30_dir

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

        print(f"[HES_CD30] {len(self.paired_files)} paires HES/CD30 trouvées dans {split}.")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, index):
        fname = self.paired_files[index]
        hes_path = os.path.join(self.hes_dir, fname)
        cd30_path = os.path.join(self.cd30_dir, fname)
        hes_img = Image.open(hes_path).convert('RGB')
        cd30_img = Image.open(cd30_path).convert('RGB')
        hes_img = self.transform(hes_img)
        cd30_img = self.transform(cd30_img)
        return hes_img, cd30_img, fname
