"""Arsitektur CNN Arun-D (dibuat dari nol, tanpa pretrained).

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


class ArunDCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def expand_classes(self, new_num_classes: int) -> None:
        """Ekspansi linear head untuk mendukung penambahan kelas baru tanpa merusak bobot lama."""
        if new_num_classes <= self.num_classes:
            return
        old_linear = self.classifier
        in_feat = old_linear.in_features
        device = old_linear.weight.device
        dtype = old_linear.weight.dtype
        new_linear = nn.Linear(in_feat, new_num_classes, device=device, dtype=dtype)
        with torch.no_grad():
            new_linear.weight[:self.num_classes] = old_linear.weight
            if old_linear.bias is not None and new_linear.bias is not None:
                new_linear.bias[:self.num_classes] = old_linear.bias
            nn.init.normal_(new_linear.weight[self.num_classes:], 0, 0.01)
            if new_linear.bias is not None:
                nn.init.zeros_(new_linear.bias[self.num_classes:])
        self.classifier = new_linear
        self.num_classes = new_num_classes


ArunaCNN = ArunDCNN


# --- Arsitektur Enhanced dengan Residual Connection & Squeeze-and-Excitation (SE) ---

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block untuk kalibrasi channel secara adaptif."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResidualSEBlock(nn.Module):
    """Blok Residual + SE untuk akurasi lebih tinggi dan representasi fitur kaya."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        out = self.relu(out + res)
        return out


class EnhancedArunCNN(nn.Module):
    """CNN modern dengan Residual SE blocks dan dynamic classification head."""
    def __init__(
        self,
        num_classes: int = 10,
        stages: tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.2,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stages[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stages[0]),
            nn.ReLU(inplace=True),
        )
        layers = []
        in_ch = stages[0]
        for idx, out_ch in enumerate(stages):
            stride = 1 if idx == 0 else 2
            layers.append(ResidualSEBlock(in_ch, out_ch, stride=stride, dropout=dropout))
            layers.append(ResidualSEBlock(out_ch, out_ch, stride=1, dropout=dropout))
            in_ch = out_ch

        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(stages[-1], num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        x = self.head_dropout(x)
        return self.classifier(x)

    def expand_classes(self, new_num_classes: int) -> None:
        """Ekspansi classifier head untuk penambahan kategori secara dinamis."""
        if new_num_classes <= self.num_classes:
            return
        old_linear = self.classifier
        in_feat = old_linear.in_features
        device = old_linear.weight.device
        dtype = old_linear.weight.dtype
        new_linear = nn.Linear(in_feat, new_num_classes, device=device, dtype=dtype)
        with torch.no_grad():
            new_linear.weight[:self.num_classes] = old_linear.weight
            if old_linear.bias is not None and new_linear.bias is not None:
                new_linear.bias[:self.num_classes] = old_linear.bias
            nn.init.normal_(new_linear.weight[self.num_classes:], 0, 0.01)
            if new_linear.bias is not None:
                nn.init.zeros_(new_linear.bias[self.num_classes:])
        self.classifier = new_linear
        self.num_classes = new_num_classes


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    arch = m.get("architecture", "arun_d")
    num_classes = m.get("num_classes", 10)

    if arch == "enhanced":
        stages = tuple(m.get("stages", m.get("channels", [64, 128, 256])))
        dropout = m.get("dropout", 0.2)
        return EnhancedArunCNN(num_classes=num_classes, stages=stages, dropout=dropout)

    # Default fallback to ArunDCNN baseline
    channels = tuple(m.get("channels", [64, 128, 256]))
    dropout = m.get("dropout", 0.3)
    return ArunDCNN(num_classes=num_classes, channels=channels, dropout=dropout)
