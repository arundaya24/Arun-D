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
    "person", "human", "man", "woman", "cowboy hat", "bonnet",
    "life jacket", "space suit", "ski mask", "baseball player"
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

        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.cfg = ckpt["config"]
        self.classes: list[str] = ckpt["classes"]
        self.version: str = ckpt.get("model_version", "unknown")
        self.model = build_model(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.transform = get_eval_transform(self.cfg)

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
                        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
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
    def detect_human(image: Image.Image, top_preds: list[dict], custom_preds: list[dict] = None) -> Tuple[bool, float, Optional[str]]:
        """Mendeteksi apakah subjek dalam citra adalah manusia berdasarkan
        kombinasi atribut manusia, pakaian, kelas custom, dan warna kulit.
        """
        human_matches = [
            item for item in top_preds
            if item["label"].lower().strip() in HUMAN_RELATED_LABELS
        ]

        if human_matches:
            match_score = human_matches[0]["score"]
            human_conf = min(0.98, max(match_score, 0.65 + match_score * 0.35))
            return True, round(human_conf, 4), human_matches[0]["label"]

        if custom_preds:
            for item in custom_preds:
                if item["label"].lower().strip() in ("manusia", "human") and item["score"] >= 0.25:
                    return True, round(max(item["score"], 0.70), 4), "manusia"

        skin_ratio = 0.0
        try:
            arr = np.array(image.convert("RGB"))
            ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
            skin_mask = (
                (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) &
                (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 127)
            )
            skin_ratio = float(np.sum(skin_mask)) / float(skin_mask.size)
        except Exception:
            skin_ratio = 0.0

        non_human_strong = False
        if top_preds:
            top_label = top_preds[0]["label"].lower()
            animal_landscape_indicators = (
                "dog", "cat", "horse", "bird", "frog", "deer", "alp", "mountain", "valley",
                "volcano", "tree", "plant", "car", "locomotive", "airplane", "ship", "truck"
            )
            if any(ind in top_label for ind in animal_landscape_indicators) and top_preds[0]["score"] >= 0.20:
                non_human_strong = True

        if skin_ratio >= 0.25 and not non_human_strong:
            human_conf = min(0.85, 0.45 + skin_ratio * 0.7)
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
        a_res = self.predict(img, top_k=top_k)

        if self.has_hybrid:
            h_res = self.predict_hybrid(img, top_k=top_k)
            custom_classes = set(self.classes[10:]) if len(self.classes) > 10 else set()
            a_top = a_res["predicted"]
            a_conf = a_res["confidence"]
            h_top = h_res["predicted"]
            h_conf = h_res["confidence"]

            # 1. Jika Arun-D mendeteksi kelas koreksi custom dengan keyakinan cukup
            if a_top in custom_classes and a_conf >= 0.18:
                base_result = a_res
            # 2. Jika Arun-D mendeteksi kelas hewan/objek kuat (misal horse > 0.60)
            elif a_top in ("horse", "frog", "deer", "cat", "dog", "bird") and a_conf >= 0.60:
                base_result = a_res
            else:
                # 3. MobileNetV3 pada 1.000 kelas ImageNet
                top2_conf = h_res["top_k"][1]["score"] if len(h_res["top_k"]) > 1 else 0.0
                margin = max(0.0, h_conf - top2_conf)
                calibrated_conf = min(0.98, max(h_conf, round(0.40 + h_conf * 0.50 + margin * 0.4, 4)))
                h_res["confidence"] = calibrated_conf
                h_res["top_k"][0]["score"] = calibrated_conf
                h_res["is_uncertain"] = calibrated_conf < 0.50
                h_res["is_low_confidence"] = h_res["is_uncertain"]
                base_result = h_res
        else:
            base_result = a_res

        is_human, human_conf, subtype = self.detect_human(img, base_result.get("top_k", []), a_res.get("top_k", []))
        base_result["is_human"] = is_human

        if is_human and subtype:
            base_result["predicted"] = "human"
            base_result["confidence"] = max(base_result["confidence"], human_conf)
            existing = [item for item in base_result.get("top_k", []) if item["label"] != "human"]
            base_result["top_k"] = [{"label": "human", "score": base_result["confidence"]}] + existing[:2]
            base_result["is_uncertain"] = base_result["confidence"] < 0.50
            base_result["is_low_confidence"] = base_result["is_uncertain"]

        return base_result
