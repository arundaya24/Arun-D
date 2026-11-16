"""Dataset CIFAR-10 dengan pembagian train / validation / test."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from .transforms import get_eval_transform, get_train_transform

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class _TransformView(Dataset):
    """Membungkus Subset agar tiap split punya transform sendiri."""

    def __init__(self, base: CIFAR10, indices: np.ndarray, transform):
        self.base, self.indices, self.transform = base, indices, transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, label = self.base[int(self.indices[i])]  # img = PIL (base tanpa transform)
        return self.transform(img), label


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Mengembalikan (train_loader, val_loader, test_loader).

    - Train+Val berasal dari 50.000 gambar train CIFAR-10 (dibagi 90/10).
    - Test = 10.000 gambar test resmi, tidak disentuh saat training.
    """
    d, t = cfg["data"], cfg["train"]

    full_train = CIFAR10(d["root"], train=True, download=True)   # tanpa transform
    test_base = CIFAR10(d["root"], train=False, download=True)

    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(full_train))
    n_val = int(len(full_train) * d["val_ratio"])
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = _TransformView(full_train, train_idx, get_train_transform(cfg))
    val_ds = _TransformView(full_train, val_idx, get_eval_transform(cfg))
    test_ds = _TransformView(test_base, np.arange(len(test_base)), get_eval_transform(cfg))

    pin = torch.cuda.is_available()
    common = dict(
        batch_size=t["batch_size"],
        num_workers=d["num_workers"],
        pin_memory=pin,
        persistent_workers=d["num_workers"] > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader
