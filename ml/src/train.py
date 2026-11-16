"""Training pipeline.

Satu iterasi training:
  1. FORWARD   : logits = model(x)
  2. LOSS      : loss = CrossEntropy(logits, y)
  3. BACKWARD  : loss.backward()  -> menghitung dLoss/dW tiap parameter (backpropagation)
  4. UPDATE    : optimizer.step() -> W <- W - lr * gradien (gradient descent / Adam)
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .dataset import CLASSES, build_dataloaders
from .evaluate import evaluate, full_report
from .model import build_model
from .utils import (count_parameters, get_device, load_config, plot_history,
                    save_json, set_seed)


def build_optimizer(model: nn.Module, t: dict) -> torch.optim.Optimizer:
    if t["optimizer"] == "adam":
        return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                               weight_decay=t["weight_decay"], nesterov=True)
    raise ValueError(f"Optimizer tidak dikenal: {t['optimizer']}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, grad_clip):
    model.train()  # aktifkan Dropout, BatchNorm pakai statistik batch
    total_loss, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, leave=False, desc="train"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)          # gradien lama dibuang
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)                          # 1. forward
            loss = criterion(logits, y)                # 2. loss
        scaler.scale(loss).backward()                  # 3. backward (skala AMP)
        if grad_clip:
            scaler.unscale_(optimizer)                 # kembalikan skala sebelum clipping
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # cegah exploding gradient
        scaler.step(optimizer)                         # 4. update bobot
        scaler.update()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def run_training(config_path: str = "ml/configs/default.yaml") -> dict:
    cfg = load_config(config_path)
    t = cfg["train"]
    set_seed(cfg["seed"])
    device = get_device()
    use_amp = bool(t["mixed_precision"]) and device.type == "cuda"
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    model = build_model(cfg).to(device)
    print(f"Parameter: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, t)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
                 if t["scheduler"] == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val, best_epoch, patience_left = 0.0, 0, t["early_stopping_patience"]
    ckpt_path = out_dir / "model.pt"
    start = time.time()

    for epoch in range(1, t["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                          device, use_amp, t["grad_clip"])
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, lr_now)):
            history[k].append(round(v, 6))
        print(f"Epoch {epoch:02d}/{t['epochs']} | loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {va_loss:.4f} val_acc {va_acc:.4f} | lr {lr_now:.5f}")

        if va_acc > best_val:   # simpan model terbaik berdasarkan validasi
            best_val, best_epoch, patience_left = va_acc, epoch, t["early_stopping_patience"]
            torch.save({"state_dict": model.state_dict(), "config": cfg, "classes": CLASSES,
                        "model_version": cfg["output"]["model_version"]}, ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping di epoch {epoch} (terbaik: epoch {best_epoch}).")
                break

    # Evaluasi akhir pada TEST set memakai model terbaik
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"\nTEST  loss {test_loss:.4f} | akurasi {test_acc:.4f}")

    report = full_report(model, test_loader, device, CLASSES, out_dir)
    metrics = {
        "model_version": cfg["output"]["model_version"],
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val, 4),
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 4),
        "parameters": count_parameters(model),
        "train_seconds": round(time.time() - start, 1),
        "history": history,
        **report,
    }
    save_json(metrics, out_dir / "metrics.json")
    save_json(CLASSES, out_dir / "classes.json")
    plot_history(history, out_dir / "training_curves.png")
    print(f"Artefak tersimpan di: {out_dir.resolve()}")
    return metrics
