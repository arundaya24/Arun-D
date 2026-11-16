"""Preprocessing & augmentation.

Aturan emas: transform untuk validation/test/inference HARUS identik.
Augmentasi hanya untuk training.
"""
from __future__ import annotations

from torchvision import transforms as T


def get_train_transform(cfg: dict) -> T.Compose:
    d = cfg["data"]
    return T.Compose(
        [
            T.RandomCrop(d["image_size"], padding=4),   # geser acak
            T.RandomHorizontalFlip(p=0.5),              # cermin kiri-kanan
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),                               # [0,255] -> [0,1], HWC -> CHW
            T.Normalize(d["mean"], d["std"]),           # rata-rata 0, std 1 per kanal
        ]
    )


def get_eval_transform(cfg: dict) -> T.Compose:
    d = cfg["data"]
    return T.Compose(
        [
            T.Resize((d["image_size"], d["image_size"])),
            T.ToTensor(),
            T.Normalize(d["mean"], d["std"]),
        ]
    )
