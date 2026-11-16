"""Evaluasi model: loss, akurasi, classification report, confusion matrix."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, criterion=None) -> tuple[float, float]:
    """Mengembalikan (rata-rata loss, akurasi). Tanpa gradien, mode eval."""
    model.eval()  # matikan Dropout, BatchNorm pakai statistik tersimpan
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def collect_predictions(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def full_report(model, loader, device, classes: list[str], out_dir: str | Path) -> dict:
    """Menyimpan confusion matrix (PNG) dan mengembalikan metrik per kelas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_pred = collect_predictions(model, loader, device)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=range(len(classes)), yticks=range(len(classes)),
        xticklabels=classes, yticklabels=classes,
        xlabel="Prediksi", ylabel="Label sebenarnya", title="Confusion Matrix (test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    return {
        "per_class": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score")}
                      for c in classes},
        "confusion_matrix": cm.tolist(),
    }
