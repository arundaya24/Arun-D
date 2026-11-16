from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.database import get_db
from backend.app.models import Prediction
from backend.app.schemas.prediction import PredictionResult
from backend.app.services.inference import get_predictor
from ml.src.errors import ImageDecodeError

router = APIRouter(prefix="/api", tags=["predict"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@router.post("/predict", response_model=PredictionResult)
def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Format tidak didukung. Gunakan JPG, PNG, WEBP, atau BMP.")

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"Ukuran file melebihi {get_settings().max_upload_mb} MB.")
    if not data:
        raise HTTPException(400, "File kosong.")

    try:
        result = predictor.predict_bytes(data, top_k=3)
    except ImageDecodeError as e:
        raise HTTPException(400, str(e)) from e

    row = Prediction(
        filename=(file.filename or "unknown")[:255],
        predicted=result["predicted"],
        confidence=result["confidence"],
        top_k_json=result["top_k"],
        inference_ms=result["inference_ms"],
        model_version=result["model_version"],
    )
    db.add(row)
    db.commit()
    return PredictionResult(id=row.id, **result)
