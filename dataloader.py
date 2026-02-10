import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class dataloader(Dataset):
    """
    Concept : fournir un accès standardise aux images d’un domaine (HES ou CD30)
    dans un format compatible PyTorch, pour alimenter un pipeline d’entraînement.
    """
    def __init__(self, root, image_size=256, split='train'):
        super().__init__()
        self.root = root
        self.image_size = image_size
        hes_dir = os.path.join(root, split, 'HES')
        cd30_dir = os.path.join(root, split, 'CD30')

        hes_files = set(os.listdir(hes_dir))
        cd30_files = set(os.listdir(cd30_dir))
        paired_files = sorted(list(hes_files & cd30_files))
        
        self.paired_files = paired_files
        self.hes_dir = hes_dir
        self.cd30_dir = cd30_dir
    
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            ])

        print(f"[HES_CD30] {len(self.paired_files)} paires HES/CD30 trouvees dans {split}.")

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
        transforms.Resize(cfg.IMAGE_SIZE), transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor()
    ]
    if cfg.RANDOM_FLIP: train_transform.insert(2, transforms.RandomHorizontalFlip())
    
    root = os.path.join(cfg.DATA_DIR, 'dataset_v4')
    init_ds = dataloader(root, image_size=cfg.IMAGE_SIZE, transform=cmp(train_transform), split='train')
    final_ds = dataloader(root, image_size=cfg.IMAGE_SIZE, transform=cmp(train_transform), split='train')
    mean_final = torch.tensor(0.)
    var_final = torch.tensor(1. * 10 ** 3)
    return init_ds, final_ds, mean_final, var_final
