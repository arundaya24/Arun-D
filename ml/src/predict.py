"""
predict.py – Dual-engine inference: ArunDCNN + HybridMobileNetV3-Small
"""

from __future__ import annotations

import json
import io
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
)

from .errors import ModelNotLoadedError, InvalidInputError
from .model import ArunDCNN
from .transforms import preprocess


class Predictor:
    """Dual-engine predictor: ArunDCNN (primary) + MobileNetV3-Small (hybrid)."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, artifacts_dir: str = "ml/artifacts", device: Optional[str] = None) -> None:
        self.artifacts_dir = Path(artifacts_dir)

        # --- device ---------------------------------------------------
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Primary model: ArunDCNN ----------------------------------
        model_path = self.artifacts_dir / "model.pt"
        if not model_path.exists():
            raise ModelNotLoadedError(f"ArunDCNN weights not found at {model_path}")

        self._model: ArunDCNN = ArunDCNN()
        state = torch.load(model_path, map_location=self.device)
        self._model.load_state_dict(state)
        self._model.to(self.device)
        self._model.eval()

        # --- Hybrid model: MobileNetV3-Small --------------------------
        self._hybrid: Optional[tv_models.MobileNetV3] = None
        self._hybrid_classes: Optional[list] = None
        self._hybrid_transform: Optional[Compose] = None

        _cache_path = Path.home() / ".cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"
        _imagenet_path = self.artifacts_dir / "imagenet_classes.json"

        if _cache_path.exists() and _imagenet_path.exists():
            try:
                with open(_imagenet_path, "r", encoding="utf-8") as f:
                    self._hybrid_classes = json.load(f)

                _net = tv_models.mobilenet_v3_small(weights=None)
                _net.load_state_dict(torch.load(_cache_path, map_location=self.device))
                _net.to(self.device)
                _net.eval()
                self._hybrid = _net

                self._hybrid_transform = Compose([
                    Resize(256),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
            except Exception:
                # Silent fallback – do not crash if hybrid loading fails.
                self._hybrid = None
                self._hybrid_classes = None
                self._hybrid_transform = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def decode(image) -> np.ndarray:
        """Decode various image inputs to a uint8 NumPy HWC array (RGB)."""
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        raise InvalidInputError(f"Unsupported image type: {type(image)}")

    # ------------------------------------------------------------------
    # ArunDCNN inference
    # ------------------------------------------------------------------

    def predict(self, image, top_k: int = 3) -> dict:
        """Run ArunDCNN inference.

        Parameters
        ----------
        image:
            numpy array (HWC, BGR or RGB) or PIL.Image.
        top_k:
            Number of top predictions to return.

        Returns
        -------
        dict with keys: label, confidence, top_k, latency_ms, model_version
        """
        rgb = self.decode(image)
        tensor = preprocess(rgb).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(0)))
        top_list = [
            {"index": int(idx), "confidence": float(prob)}
            for prob, idx in zip(top_probs, top_indices)
        ]

        best = top_list[0]
        return {
            "label": best["index"],
            "confidence": best["confidence"],
            "top_k": top_list,
            "latency_ms": round(latency_ms, 2),
            "model_version": "arun-d-cnn",
        }

    # ------------------------------------------------------------------
    # Hybrid (MobileNetV3-Small) inference
    # ------------------------------------------------------------------

    def predict_hybrid(self, image, top_k: int = 3) -> dict:
        """Run HybridMobileNetV3-Small inference.

        Parameters
        ----------
        image:
            numpy array (HWC) or PIL.Image.
        top_k:
            Number of top predictions to return.

        Returns
        -------
        Same dict format as predict().
        """
        if self._hybrid is None:
            raise ModelNotLoadedError("Hybrid MobileNetV3-Small model is not available.")

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise InvalidInputError(f"Unsupported image type for hybrid: {type(image)}")

        tensor = self._hybrid_transform(pil_image).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self._hybrid(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(0)))
        top_list = [
            {
                "index": int(idx),
                "label": (
                    self._hybrid_classes[int(idx)]
                    if self._hybrid_classes and int(idx) < len(self._hybrid_classes)
                    else int(idx)
                ),
                "confidence": float(prob),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        best = top_list[0]
        return {
            "label": best["label"],
            "confidence": best["confidence"],
            "top_k": top_list,
            "latency_ms": round(latency_ms, 2),
            "model_version": "hybrid-mobilenet-v3s",
        }

    # ------------------------------------------------------------------
    # Bytes entry-point (dual-engine routing)
    # ------------------------------------------------------------------

    def predict_bytes(self, data: bytes, top_k: int = 3) -> dict:
        """Decode raw image bytes and route through dual-engine inference.

        The hybrid model wins when its confidence >= 0.60; otherwise the
        ArunDCNN result is used.  If the hybrid model is unavailable the
        ArunDCNN result is always used.

        Parameters
        ----------
        data:
            Raw image bytes (JPEG, PNG, …).
        top_k:
            Number of top predictions to return.

        Returns
        -------
        Merged result dict that includes:
            - All keys from the winning predict() / predict_hybrid() call
            - engine_used   (str)  : "hybrid_mobilenet" | "arun_d_cnn"
            - is_low_confidence (bool): True when winning confidence < 0.60
        """
        # --- Decode bytes to PIL image --------------------------------
        try:
            pil_image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise InvalidInputError(f"Cannot decode image bytes: {exc}") from exc

        # --- ArunDCNN (always runs) -----------------------------------
        rgb_array = np.array(pil_image)
        arun_result = self.predict(rgb_array, top_k=top_k)

        # --- Dual-engine routing -------------------------------------
        CONFIDENCE_THRESHOLD = 0.60

        if self._hybrid is not None:
            hybrid_result = self.predict_hybrid(pil_image, top_k=top_k)

            if hybrid_result["confidence"] >= CONFIDENCE_THRESHOLD:
                winning_result = hybrid_result
                engine_used = "hybrid_mobilenet"
            else:
                winning_result = arun_result
                engine_used = "arun_d_cnn"
        else:
            winning_result = arun_result
            engine_used = "arun_d_cnn"

        # --- Build merged result -------------------------------------
        result = dict(winning_result)
        result["engine_used"] = engine_used
        result["is_low_confidence"] = winning_result["confidence"] < CONFIDENCE_THRESHOLD

        return result
