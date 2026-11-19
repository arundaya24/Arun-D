"""Inference: Dual-engine predictor (Arun-D CNN + Hybrid MobileNetV3) dengan deteksi manusia."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .errors import ImageDecodeError
from .model import build_model
from .transforms import get_eval_transform

HUMAN_RELATED_LABELS = {
    "ballplayer", "groom", "scuba diver", "suit", "academic gown",
    "trench coat", "jersey", "swimming trunks", "wig", "sombrero",
    "military uniform", "cloak", "jean", "sweatshirt", "bikini",
    "lab coat", "mortarboard", "poncho", "stole", "tights", "breastplate",
    "pajamas", "cardigan", "miniskirt", "hoopskirt", "overskirt",
    "person", "human", "man", "woman"
}


class Predictor:
    def __init__(self, artifacts_dir: str | Path = "ml/artifacts", device: str | None = None):
        self.artifacts_dir = Path(artifacts_dir)
        path = self.artifacts_dir / "model.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} tidak ditemukan. Latih model dulu: python -m ml.scripts.run_train"
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 1. Muat Arun-D Primary Model
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.cfg = ckpt["config"]
        self.classes: list[str] = ckpt["classes"]
        self.version: str = ckpt.get("model_version", "unknown")
        self.model = build_model(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.transform = get_eval_transform(self.cfg)

        # 2. Inisialisasi Hybrid Model (MobileNetV3) jika bobot & kelas tersedia
        self._hybrid_model = None
        self._hybrid_classes: list[str] = []
        self._hybrid_transform = None
        self._init_hybrid_engine()

    def _init_hybrid_engine(self) -> None:
        """Mencoba memuat MobileNetV3-Small dari cache lokal jika file bobot telah selesai diunduh."""
        cache_file = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
        classes_file = self.artifacts_dir / "imagenet_classes.json"

        if cache_file.exists() and classes_file.exists():
            try:
                if cache_file.stat().st_size > 9_000_000:
                    import torchvision.models as tv_models
                    net = tv_models.mobilenet_v3_small(weights=None)
                    state = torch.load(cache_file, map_location=self.device, weights_only=True)
                    net.load_state_dict(state)
                    net.to(self.device).eval()

                    with open(classes_file, "r", encoding="utf-8") as f:
                        self._hybrid_classes = json.load(f)

                    self._hybrid_transform = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    self._hybrid_model = net
            except Exception:
                self._hybrid_model = None

    @property
    def has_hybrid(self) -> bool:
        return self._hybrid_model is not None

    @staticmethod
    def detect_human(image: Image.Image, top_preds: list[dict]) -> Tuple[bool, float, Optional[str]]:
        """Mendeteksi apakah subjek dalam citra adalah manusia berdasarkan

        kombinasi segmentasi warna kulit (YCrCb) dan klasifikasi kelas pakaian/atribut manusia.
        """
        human_matches = [
            item for item in top_preds
            if item["label"].lower().strip() in HUMAN_RELATED_LABELS
        ]

        skin_ratio = 0.0
        try:
            arr = np.array(image.convert("RGB"))
            ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
            # Rentang warna kulit standar YCrCb: Cr [133..173], Cb [77..127]
            skin_mask = (
                (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) &
                (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 127)
            )
            skin_ratio = float(np.sum(skin_mask)) / float(skin_mask.size)
        except Exception:
            skin_ratio = 0.0

        if human_matches:
            match_score = human_matches[0]["score"]
            human_conf = min(0.98, max(match_score, 0.55 + skin_ratio * 0.7))
            return True, round(human_conf, 4), human_matches[0]["label"]

        if skin_ratio >= 0.14:
            human_conf = min(0.92, 0.45 + skin_ratio * 0.85)
            return True, round(human_conf, 4), None

        return False, 0.0, None

    @staticmethod
    def decode(data: bytes) -> Image.Image:
        """Bytes -> PIL RGB."""
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise ImageDecodeError("File bukan gambar yang valid.") from e

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        t0 = time.perf_counter()
        x = self.transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        k = min(top_k, len(self.classes))
        scores, idx = probs.topk(k)
        top = [
            {"label": self.classes[i], "score": round(s, 4)}
            for s, i in zip(scores.tolist(), idx.tolist())
        ]
        conf = top[0]["score"]
        is_uncertain = conf < 0.60
        return {
            "predicted": top[0]["label"],
            "confidence": conf,
            "top_k": top,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
            "model_version": self.version,
            "engine_used": "arun_d_cnn",
            "is_uncertain": is_uncertain,
            "is_low_confidence": is_uncertain,
            "is_human": False,
        }

    @torch.no_grad()
    def predict_hybrid(self, image: Image.Image, top_k: int = 3) -> dict:
        if self._hybrid_model is None or self._hybrid_transform is None:
            return self.predict(image, top_k)

        t0 = time.perf_counter()
        x = self._hybrid_transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self._hybrid_model(x), dim=1)[0]
        k = min(top_k, len(self._hybrid_classes))
        scores, idx = probs.topk(k)
        top = [
            {
                "label": self._hybrid_classes[i] if i < len(self._hybrid_classes) else str(i),
                "score": round(s, 4),
            }
            for s, i in zip(scores.tolist(), idx.tolist())
        ]
        conf = top[0]["score"]
        is_uncertain = conf < 0.60
        return {
            "predicted": top[0]["label"],
            "confidence": conf,
            "top_k": top,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
            "model_version": "hybrid-mobilenet-v3s",
            "engine_used": "hybrid_mobilenet",
            "is_uncertain": is_uncertain,
            "is_low_confidence": is_uncertain,
            "is_human": False,
        }

    def predict_bytes(self, data: bytes, top_k: int = 3) -> dict:
        img = self.decode(data)
        if self.has_hybrid:
            h_res = self.predict_hybrid(img, top_k)
            if h_res["confidence"] >= 0.60:
                base_result = h_res
            else:
                a_res = self.predict(img, top_k)
                base_result = a_res if a_res["confidence"] >= h_res["confidence"] else h_res
        else:
            base_result = self.predict(img, top_k)

        # Evaluasi deteksi manusia
        is_human, human_conf, subtype = self.detect_human(img, base_result.get("top_k", []))
        base_result["is_human"] = is_human

        if is_human:
            # Jika terdeteksi manusia dengan sinyal kuat atau model kurang yakin pada objek lain
            if subtype or human_conf >= 0.65 or base_result["confidence"] < 0.60:
                old_label = base_result["predicted"]
                base_result["predicted"] = "human"
                base_result["confidence"] = max(base_result["confidence"], human_conf)
                # Perbarui daftar top_k
                existing = [item for item in base_result.get("top_k", []) if item["label"] != "human"]
                base_result["top_k"] = [{"label": "human", "score": human_conf}] + existing[:2]
                base_result["is_uncertain"] = base_result["confidence"] < 0.60
                base_result["is_low_confidence"] = base_result["is_uncertain"]

        return base_result
