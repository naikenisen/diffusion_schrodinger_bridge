"""
Ce fichier définit le dataset PyTorch HES_CD30.
Il sert à charger des images depuis un dossier organisé par domaines (HES ou CD30),
à appliquer les transformations (resize, crop, conversion en tenseur), puis à fournir
les images au DataLoader pendant l’entraînement. Il ne fait pas l’entraînement : il ne fait
que "lire et préparer" les images.
"""


import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class HES_CD30(Dataset):
    """
    Cette classe permet de charger des images médicales
    pour les utiliser dans un modèle PyTorch.

    Elle lit des images stockées sur le disque et les transforme
    en tenseurs exploitables par un réseau de neurones.

    Même si la documentation parle d’images HES et CD30 appariées,
    cette classe charge UN SEUL type d’image à la fois
    (soit HES, soit CD30).
    """
        
    """Dataset for paired HES / CD30 virtual staining.
    
    Expected folder structure:
        root/
        ├── HES/
        │   ├── c/
        │   │   ├── patch_x2000_y32000.jpg
        │   │   └── ...
        │   ├── e/
        │   └── ...
        └── CD30/
            ├── c/
            │   ├── patch_x2000_y32000.jpg
            │   └── ...
            ├── e/
            └── ...
    
    Each HES image has a paired CD30 image with the same subfolder
    and filename.
    """

    def __init__(self, root, image_size=256, domain='HES', transform=None):
        """
        Args:
            root: path to dataset_v2/ directory
            image_size: resize images to this size
            domain: 'HES' or 'CD30'
            transform: optional torchvision transform (overrides default)
        """
        super().__init__()
         # Sauvegarde des paramètres
        self.root = root
        self.domain = domain
        self.image_size = image_size

        # Construction du chemin vers le dossier des images
        # Exemple : root/HES/ ou root/CD30/
        domain_dir = os.path.join(root, domain)
        # Recherche de toutes les images .jpg dans les sous-dossiers
        # (par exemple c/, e/, etc.)
        self.image_paths = sorted(
            glob.glob(os.path.join(domain_dir, '*', '*.jpg')))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {domain_dir}. "
                f"Expected structure: {domain_dir}/<subfolder>/*.jpg"
            )
        
        # Définition des transformations appliquées aux images
        if transform is not None:
            # Si l’utilisateur fournit ses propres transformations    
            self.transform = transform
        else:
            # Transformations par défaut :
            # - redimensionnement
            # - découpe centrale
            # - conversion en tenseur PyTorch
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
        # Message informatif pour vérifier que le chargement s’est bien passé
        print(f"[HES_CD30] Loaded {len(self.image_paths)} images from {domain_dir}")

    def __len__(self):
        # Retourne le nombre total d’images du dataset.
        # PyTorch utilise cette information pour parcourir les données.
        return len(self.image_paths)

    def __getitem__(self, index):
        # Récupère le chemin de l’image demandée
        img_path = self.image_paths[index]
        # Ouvre l’image et la convertit en RGB (3 canaux)
        img = Image.open(img_path).convert('RGB')
        # Applique les transformations définies plus haut
        img = self.transform(img)
        # PyTorch attend souvent un couple (image, label).
        # Ici, il n’y a pas de label, donc on renvoie une valeur factice.
        return img, 0  # label fictif pour compatibilité
