import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from ml.config import CLASSES, PROJECT_ROOT, SPLITS_DIR
from ml.preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess

CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}


def read_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS_DIR / f"{name}.csv")


def train_augmentation() -> v2.Compose:
    return v2.Compose([
        v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ])


class MRIDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        frame = read_split(split)
        self.paths = frame["path"].tolist()
        self.targets = frame["label"].map(CLASS_TO_INDEX).to_numpy()
        self.augment = train_augmentation() if augment else None
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        gray = preprocess(PROJECT_ROOT / self.paths[idx])
        x = torch.from_numpy(np.asarray(gray, dtype=np.uint8).copy()).unsqueeze(0)
        if self.augment is not None:
            x = self.augment(x)
        x = x.float().div(255.0).expand(3, -1, -1)
        return (x - self.mean) / self.std, int(self.targets[idx])
