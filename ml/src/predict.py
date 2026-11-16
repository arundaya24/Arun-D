"""Inference: memuat model terlatih dan memprediksi satu gambar."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .errors import ImageDecodeError
from .model import build_model
from .transforms import get_eval_transform


class Predictor:
    def __init__(self, artifacts_dir: str | Path = "ml/artifacts", device: str | None = None):
        path = Path(artifacts_dir) / "model.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} tidak ditemukan. Latih model dulu: python -m ml.scripts.run_train"
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.cfg = ckpt["config"]
        self.classes: list[str] = ckpt["classes"]
        self.version: str = ckpt.get("model_version", "unknown")
        self.model = build_model(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.transform = get_eval_transform(self.cfg)  # SAMA dengan validation/test

    @staticmethod
    def decode(data: bytes) -> Image.Image:
        """Bytes -> PIL RGB, memakai OpenCV (BGR -> RGB)."""
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ImageDecodeError("File bukan gambar yang valid.")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        t0 = time.perf_counter()
        x = self.transform(image).unsqueeze(0).to(self.device)   # (1,3,H,W)
        probs = torch.softmax(self.model(x), dim=1)[0]           # logits -> probabilitas
        k = min(top_k, len(self.classes))
        scores, idx = probs.topk(k)
        top = [{"label": self.classes[i], "score": round(s, 4)}
               for s, i in zip(scores.tolist(), idx.tolist())]
        return {
            "predicted": top[0]["label"],
            "confidence": top[0]["score"],
            "top_k": top,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
            "model_version": self.version,
        }

    def predict_bytes(self, data: bytes, top_k: int = 3) -> dict:
        return self.predict(self.decode(data), top_k)
