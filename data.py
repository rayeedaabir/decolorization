"""Datasets. Applies a decolorizer on the fly, so you never pre-export grays.

Expects an ImageFolder layout (one subfolder per class), which is how UC Merced
and AID ship:
    data/UCM/agricultural/*.tif
    data/UCM/airplane/*.tif ...

Requires torch + torchvision (installed on your training machine).
"""
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class DecolorizedDataset(Dataset):
    """ImageFolder + decolorizer(RGB->gray), gray replicated to 3 channels."""

    def __init__(self, root, decolorizer, img_size=224, train=True):
        self.base = ImageFolder(root)
        self.classes = self.base.classes
        self.decolorizer = decolorizer
        if train:
            geom = [transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                    transforms.RandomHorizontalFlip()]
        else:
            geom = [transforms.Resize(int(img_size * 1.14)),
                    transforms.CenterCrop(img_size)]
        self.geom = transforms.Compose(geom)
        self.norm = transforms.Normalize(_MEAN, _STD)

    def __len__(self):
        return len(self.base.samples)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        pil = self.geom(Image.open(path).convert("RGB"))
        rgb = np.asarray(pil, np.float32) / 255.0
        gray = np.clip(np.asarray(self.decolorizer(rgb)), 0, 1).astype(np.float32)
        t = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1)   # 3 x H x W
        return self.norm(t), label


def make_splits(root, decolorizer, img_size=224, val_split=0.2, seed=0):
    """Deterministic train/val split with train-time aug vs eval transforms."""
    train_full = DecolorizedDataset(root, decolorizer, img_size, train=True)
    val_full = DecolorizedDataset(root, decolorizer, img_size, train=False)
    n = len(train_full)
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - val_split))
    tr, va = idx[:cut].tolist(), idx[cut:].tolist()
    return Subset(train_full, tr), Subset(val_full, va), train_full.classes
