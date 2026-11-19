"""Active Learning & Human-in-the-Loop: fine-tuning model dari koreksi user."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .model import build_model
from .utils import load_config


def get_correction_summary(corrections_dir: Union[str, Path] = "ml/data/corrections") -> Dict:
    """Menghitung jumlah data koreksi per kelas label."""
    corrections_dir = Path(corrections_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    by_class: Dict[str, int] = {}
    total = 0

    if not corrections_dir.exists():
        return {"total_images": 0, "by_class": {}}

    for class_dir in sorted(corrections_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        count = sum(
            1 for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_exts
        )
        if count > 0:
            by_class[class_dir.name] = count
            total += count

    return {"total_images": total, "by_class": by_class}


class CorrectionDataset(Dataset):
    """Dataset yang memuat gambar hasil koreksi dari pengguna."""

    def __init__(self, corrections_dir: Path, classes: List[str], transform):
        self.samples: List[Tuple[Path, int]] = []
        self.transform = transform
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for class_dir in corrections_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name not in class_to_idx:
                continue
            idx = class_to_idx[class_dir.name]
            for img_file in class_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in image_exts:
                    self.samples.append((img_file, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            tensor = self.transform(rgb_img)
        return tensor, label


def fine_tune_from_corrections(
    corrections_dir: Union[str, Path] = "ml/data/corrections",
    artifacts_dir: Union[str, Path] = "ml/artifacts",
    epochs: int = 5,
    lr: float = 1e-4,
    device: Optional[str] = None,
) -> dict:
    """Melatih ulang model secara aman (freeze feature layers, latih classification head)

    mencegah catastrophic forgetting dengan mempertahankan bobot ekstraktor fitur.
    """
    corrections_dir = Path(corrections_dir)
    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model utama tidak ditemukan di {model_path}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Muat checkpoint yang ada
    ckpt = torch.load(model_path, map_location=dev, weights_only=False)
    cfg = ckpt["config"]
    classes: List[str] = list(ckpt["classes"])

    # 2. Cek label baru di direktori koreksi
    summary = get_correction_summary(corrections_dir)
    if summary["total_images"] == 0:
        return {
            "trained_on": 0,
            "new_classes": [],
            "message": "Tidak ada data koreksi untuk retraining.",
        }

    existing_classes_lower = {c.lower() for c in classes}
    new_classes = [c for c in summary["by_class"] if c.lower() not in existing_classes_lower]
    if new_classes:
        classes.extend(new_classes)

    # 3. Bangun model dan muat bobot lama
    model = build_model(cfg).to(dev)
    # Jika jumlah kelas bertambah, ekspansi head
    if len(classes) > model.num_classes:
        model.load_state_dict(ckpt["state_dict"])
        model.expand_classes(len(classes))
    else:
        model.load_state_dict(ckpt["state_dict"])

    # Pastikan seluruh layer dan bobot berada pada target device
    model.to(dev)

    # 4. Freeze feature extraction layers agar bobot representasi tidak rusak
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    # 5. Siapkan Dataset & DataLoader
    img_size = cfg.get("data", {}).get("image_size", 32)
    mean = cfg.get("data", {}).get("mean", [0.4914, 0.4822, 0.4465])
    std = cfg.get("data", {}).get("std", [0.2470, 0.2435, 0.2616])

    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    dataset = CorrectionDataset(corrections_dir, classes, train_transform)
    if len(dataset) == 0:
        return {"trained_on": 0, "new_classes": new_classes, "message": "Tidak ada data valid."}

    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

    # 6. Training loop
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss().to(dev)

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        final_loss = epoch_loss / len(dataset)

    # 7. Simpan model yang diperbarui
    ver_base = ckpt.get("model_version", "arun-d")
    new_version = f"{ver_base}-ft"
    cfg["model"]["num_classes"] = len(classes)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "classes": classes,
            "model_version": new_version,
        },
        model_path,
    )

    # Update juga classes.json
    classes_json_path = artifacts_dir / "classes.json"
    with open(classes_json_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    return {
        "trained_on": len(dataset),
        "new_classes": new_classes,
        "epochs": epochs,
        "final_loss": round(final_loss, 4),
        "updated_classes": classes,
        "model_version": new_version,
    }
