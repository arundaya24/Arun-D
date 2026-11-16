"""Menyimpan satu instance Predictor (model dimuat sekali, bukan tiap request)."""
from __future__ import annotations

import threading

from fastapi import HTTPException

from backend.app.core.config import get_settings

_predictor = None
_lock = threading.Lock()


def get_predictor():
    """Dependency FastAPI. Di-override pada tes agar tidak butuh PyTorch."""
    global _predictor
    if _predictor is None:
        with _lock:
            if _predictor is None:
                try:
                    from ml.src.predict import Predictor  # impor lambat: torch berat

                    _predictor = Predictor(get_settings().model_dir)
                except FileNotFoundError as e:
                    raise HTTPException(status_code=503, detail=str(e)) from e
    return _predictor
