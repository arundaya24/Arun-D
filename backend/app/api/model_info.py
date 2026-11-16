import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.core.config import get_settings

router = APIRouter(prefix="/api/model", tags=["model"])


def _metrics() -> dict:
    path = Path(get_settings().model_dir) / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "Model belum dilatih. Jalankan: python -m ml.scripts.run_train")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/info")
def info():
    m = _metrics()
    classes_path = Path(get_settings().model_dir) / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8")) if classes_path.exists() else []
    keys = ("model_version", "test_acc", "test_loss", "best_val_acc", "best_epoch",
            "epochs_run", "parameters", "train_seconds")
    return {**{k: m.get(k) for k in keys}, "classes": classes}


@router.get("/metrics")
def metrics():
    """Kurva loss/akurasi + metrik per kelas + confusion matrix untuk dashboard."""
    m = _metrics()
    return {k: m.get(k) for k in ("history", "per_class", "confusion_matrix", "test_acc", "test_loss")}
