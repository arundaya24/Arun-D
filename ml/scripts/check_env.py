"""Verifikasi environment: python -m ml.scripts.check_env"""
import sys

import torch


def main() -> None:
    print(f"Python  : {sys.version.split()[0]}")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU     : {p.name} | VRAM {p.total_memory / 1024**3:.1f} GB")
        x = torch.randn(1000, 1000, device="cuda")
        print(f"Tes matmul GPU OK: {(x @ x).sum().item():.2f}")
    else:
        print("GPU tidak terdeteksi. Pasang PyTorch versi CUDA (lihat README langkah 3).")


if __name__ == "__main__":
    main()
