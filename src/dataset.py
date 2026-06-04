import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from src.utils import NIH_LABELS


class NIHChestDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, subset=1.0):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_path)

        # Filter to only images that actually exist on disk
        print('Filtering to available images...')
        available = set(os.listdir(img_dir))
        df = df[df['Image Index'].isin(available)].reset_index(drop=True)
        print(f'Available: {len(df):,} / {len(pd.read_csv(csv_path)):,}')

        if subset < 1.0:
            df = df.sample(frac=subset, random_state=42).reset_index(drop=True)

        self.image_names = df['Image Index'].values
        self.label_matrix = self._encode_labels(df['Finding Labels'].values)

    def _encode_labels(self, raw_labels):
        matrix = np.zeros((len(raw_labels), len(NIH_LABELS)), dtype=np.float32)
        for i, label_str in enumerate(raw_labels):
            for label in label_str.split("|"):
                label = label.strip()
                if label in NIH_LABELS:
                    matrix[i, NIH_LABELS.index(label)] = 1.0
        return matrix

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_matrix[idx], self.image_names[idx]


class CheXpertDataset(Dataset):
    """
    CheXpert dataset used as OOD evaluation set.
    Maps CheXpert labels to NIH label space where possible.
    Uncertain labels (-1) treated as negative (0).

    Expected layout in Google Drive:
        data/chexpert/
            valid.csv
            PNG_valid_files/
                patient64541/
                    study1/
                        view1_frontal.png
    """

    CHEXPERT_TO_NIH = {
        'Atelectasis'     : 'Atelectasis',
        'Cardiomegaly'    : 'Cardiomegaly',
        'Pleural Effusion': 'Effusion',
        'Pneumonia'       : 'Pneumonia',
        'Pneumothorax'    : 'Pneumothorax',
        'Consolidation'   : 'Consolidation',
        'Edema'           : 'Edema',
    }

    def __init__(self, csv_path, img_root, transform=None):
        self.img_root  = img_root
        self.transform = transform
        self.labels    = NIH_LABELS

        df = pd.read_csv(csv_path)

        # Keep frontal views only
        if 'Frontal/Lateral' in df.columns:
            df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

        # Strip path prefix to get patient/study/image
        # CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg
        # → patient64541/study1/view1_frontal.jpg
        df['local_path'] = df['Path'].apply(
            lambda p: '/'.join(p.split('/')[-3:])
        )

        # Change .jpg to .png since our files are PNG
        df['local_path'] = df['local_path'].str.replace('.jpg', '.png', regex=False)

        # Filter to only existing images
        df['full_path'] = df['local_path'].apply(
            lambda p: os.path.join(img_root, p)
        )
        df = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)
        print(f'CheXpert available: {len(df)} images')

        self.paths        = df['local_path'].values
        self.label_matrix = self._encode_labels(df)

    def _encode_labels(self, df):
        matrix = np.zeros((len(df), len(self.labels)), dtype=np.float32)
        for chex_col, nih_col in self.CHEXPERT_TO_NIH.items():
            if chex_col in df.columns and nih_col in self.labels:
                idx  = self.labels.index(nih_col)
                vals = df[chex_col].fillna(0).values
                vals = np.where(vals == -1, 0, vals)
                matrix[:, idx] = vals.astype(np.float32)
        return matrix

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.paths[idx])
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.label_matrix[idx], self.paths[idx]


def get_transforms(split="train", image_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_nih_loaders(csv_path, img_dir, batch_size=32,
                    train_split=0.8, image_size=224,
                    num_workers=0, subset=1.0):
    full_dataset = NIHChestDataset(csv_path, img_dir, subset=subset)
    n = len(full_dataset)
    n_train = int(n * train_split)

    train_set = NIHChestDataset(csv_path, img_dir,
                                transform=get_transforms("train", image_size),
                                subset=subset)
    val_set   = NIHChestDataset(csv_path, img_dir,
                                transform=get_transforms("val", image_size),
                                subset=subset)

    train_loader = DataLoader(Subset(train_set, range(n_train)),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(Subset(val_set, range(n_train, n)),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader
