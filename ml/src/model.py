"""Arsitektur CNN ARUNA (dibuat dari nol, tanpa pretrained).

Alur forward propagation untuk 1 gambar 3x32x32:

  Blok 1: [Conv3x3 -> BN -> ReLU] x2 -> MaxPool -> Dropout   (3x32x32  -> 64x16x16)
  Blok 2: [Conv3x3 -> BN -> ReLU] x2 -> MaxPool -> Dropout   (64x16x16 -> 128x8x8)
  Blok 3: [Conv3x3 -> BN -> ReLU] x2 -> MaxPool -> Dropout   (128x8x8  -> 256x4x4)
  Global Average Pooling                                     (256x4x4  -> 256)
  Linear                                                     (256      -> num_classes)

Output adalah LOGITS (skor mentah). Softmax TIDAK dipasang di dalam model
karena nn.CrossEntropyLoss sudah menggabungkan log-softmax + NLL secara stabil.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )


class ArunaCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        blocks, prev = [], in_channels
        for ch in channels:
            blocks.append(conv_block(prev, ch, dropout))
            prev = ch
        self.features = nn.Sequential(*blocks)       # ekstraksi fitur
        self.pool = nn.AdaptiveAvgPool2d(1)          # rata-rata tiap peta fitur
        self.classifier = nn.Linear(prev, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init: cocok untuk ReLU, menjaga varians aktivasi tetap stabil."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(cfg: dict) -> ArunaCNN:
    m = cfg["model"]
    return ArunaCNN(
        num_classes=m["num_classes"],
        channels=tuple(m["channels"]),
        dropout=m["dropout"],
    )
