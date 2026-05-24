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
        if subset < 1.0:
            df = df.sample(frac=subset, random_state=42).reset_index(drop=True)

        self.image_names = df["Image Index"].values
        self.label_matrix = self._encode_labels(df["Finding Labels"].values)

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
                    num_workers=2, subset=1.0):
    # Use subset for local CPU testing (e.g. subset=0.01 = 1% of data)
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