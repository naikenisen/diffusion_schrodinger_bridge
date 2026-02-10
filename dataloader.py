import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import config as cfg

cmp = lambda x: transforms.Compose([*x])

class dataloader(Dataset):
    """
    Dataset apparié HES/CD30.
    - On construit la liste des fichiers communs (paired_files) une seule fois.
    - Ensuite, selon `domain`, le dataset retourne uniquement HES ou uniquement CD30,
      MAIS en conservant exactement le même ordre d'index -> appariement garanti.
    """
    def __init__(self, root, image_size=256, split='train', transform=None, domain='HES'):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.transform = transform
        self.domain = domain.upper()

        hes_dir = os.path.join(root, split, 'HES')
        cd30_dir = os.path.join(root, split, 'CD30')

        if not os.path.isdir(hes_dir):
            raise FileNotFoundError(f"Dossier introuvable: {hes_dir}")
        if not os.path.isdir(cd30_dir):
            raise FileNotFoundError(f"Dossier introuvable: {cd30_dir}")

        hes_files = set(os.listdir(hes_dir))
        cd30_files = set(os.listdir(cd30_dir))
        paired_files = sorted(list(hes_files & cd30_files))

        self.paired_files = paired_files
        self.hes_dir = hes_dir
        self.cd30_dir = cd30_dir

        if self.domain not in ("HES", "CD30"):
            raise ValueError("`domain` doit être 'HES' ou 'CD30'.")

        print(f"[HES_CD30] {len(self.paired_files)} paires HES/CD30 trouvees dans {split} (domain={self.domain}).")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, index):
        fname = self.paired_files[index]

        if self.domain == "HES":
            img_path = os.path.join(self.hes_dir, fname)
        else:  # "CD30"
            img_path = os.path.join(self.cd30_dir, fname)

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        else:
            # fallback minimal si aucun transform n'est fourni
            img = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])(img)

        return img, fname


def get_datasets():
    """
    Charge deux datasets alignés (appariés par index) :
    - init_ds  : domaine initial (HES)
    - final_ds : domaine final (CD30)

    Le même index i correspond au même fichier (fname) dans les deux datasets.
    """
    train_transform = [
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor()
    ]

    root = os.path.join(cfg.DATA_DIR, 'dataset_v4')

    # IMPORTANT: mêmes root/split -> même paired_files trié -> appariement par index garanti
    init_ds = dataloader(
        root,
        image_size=cfg.IMAGE_SIZE,
        transform=cmp(train_transform),
        split='train',
        domain='HES'
    )
    final_ds = dataloader(
        root,
        image_size=cfg.IMAGE_SIZE,
        transform=cmp(train_transform),
        split='train',
        domain='CD30'
    )

    mean_final = torch.tensor(0.)
    var_final = torch.tensor(1. * 10 ** 3)
    return init_ds, final_ds, mean_final, var_final
