import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import config as cfg

class PairedDataset(Dataset):
    def __init__(self, root, split='train', domain='HES', transform=None):
        super().__init__()
        self.transform = transform
        self.domain = domain.upper()
        
        base_dir = os.path.join(root, split)
        hes_dir = os.path.join(base_dir, 'HES')
        cd30_dir = os.path.join(base_dir, 'CD30')

        if not os.path.exists(hes_dir) or not os.path.exists(cd30_dir):
            raise FileNotFoundError(f"Dataset directories not found in {base_dir}")

        common_files = sorted(list(set(os.listdir(hes_dir)) & set(os.listdir(cd30_dir))))
        self.files = common_files
        self.source_dir = hes_dir if self.domain == 'HES' else cd30_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]
        img_path = os.path.join(self.source_dir, fname)
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, fname

def get_datasets():
    transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    root = 'dataset_v4'

    init_ds = PairedDataset(root, split='train', domain='HES', transform=transform)
    final_ds = PairedDataset(root, split='train', domain='CD30', transform=transform)

    mean_final = torch.tensor(0.)
    var_final = torch.tensor(1000.)

    return init_ds, final_ds, mean_final, var_final

def get_test_dataloader():
    transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    root = 'dataset_v4'
    dataset = PairedDataset(root, split='test', domain='HES', transform=transform)

    return DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )