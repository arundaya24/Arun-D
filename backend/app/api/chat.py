import os
import shutil
import threading
import concurrent.futures
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.database import get_db
from backend.app.models import Prediction, Feedback
from backend.app.schemas.chat import ChatBotResponse, FeedbackRequest, FeedbackResponse, RetrainStatus
from backend.app.services.inference import get_predictor
from ml.src.errors import ImageDecodeError

router = APIRouter(prefix="/api", tags=["chat"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif", "image/tiff"}

_retrain_status: dict = {
    "status": "idle",
    "message": "Belum ada retraining",
    "trained_on": None,
    "new_classes": None,
}
_retrain_lock = threading.Lock()


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatBotResponse)
async def chat(
    file: UploadFile = File(None),
    message: str = Form(""),
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
):
    settings = get_settings()

    # ── Branch 1: image provided and content-type is allowed ──────────────
    if file is not None and file.content_type in ALLOWED_TYPES:
        try:
            data = await file.read()
            result = predictor.predict_bytes(data, top_k=3)
        except ImageDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Gambar tidak dapat dibaca: {exc}") from exc

        predicted: str = result["label"]
        confidence: float = result["confidence"]
        top_k: list = result.get("top_k", [])
        confidence_pct = round(confidence * 100, 1)

        # Persist prediction to DB
        pred_row = Prediction(
            predicted_label=predicted,
            confidence=confidence,
            top_k=top_k,
            image_path=result.get("image_path"),
        )
        db.add(pred_row)
        db.commit()
        db.refresh(pred_row)

        # Build chatbot message based on confidence threshold
        if confidence >= 0.60:
            chatbot_message = (
                f"Saya cukup yakin ini adalah **{predicted}** "
                f"(keyakinan: {confidence_pct}%). "
                "Ada yang ingin dikonfirmasi atau dikoreksi?"
            )
        else:
            chatbot_message = (
                f"Saya kurang yakin, tapi mungkin ini **{predicted}** "
                f"({confidence_pct}%). "
                "Bisa tolong konfirmasi apakah prediksi ini benar?"
            )

        return ChatBotResponse(
            id=pred_row.id,
            predicted_label=predicted,
            confidence=confidence,
            confidence_pct=confidence_pct,
            top_k=top_k,
            image_path=pred_row.image_path,
            chatbot_message=chatbot_message,
        )

    # ── Branch 2: no file or disallowed type – text-only response ─────────
    msg_lower = message.lower()

    if any(kw in msg_lower for kw in ("gambar apa", "apa ini", "ini apa", "apa gambar", "tebak")):
        msg = "Silakan unggah gambar terlebih dahulu, lalu saya akan menganalisisnya! 📷"
    elif any(kw in msg_lower for kw in ("halo", "hai", "hello")):
        msg = (
            "Halo! Saya siap membantu mengklasifikasikan gambar Anda. "
            "Unggah gambar dan saya akan memberitahu apa isinya!"
        )
    elif any(kw in msg_lower for kw in ("terima kasih", "makasih")):
        msg = "Sama-sama! Jika ada gambar lain yang ingin dianalisis, silakan unggah kapan saja. 😊"
    else:
        msg = (
            "Saya adalah asisten klasifikasi gambar. "
            "Unggah gambar untuk saya analisis, atau tanyakan sesuatu tentang hasil prediksi!"
        )

    return ChatBotResponse(chatbot_message=msg)


# ---------------------------------------------------------------------------
# POST /api/feedback
# ---------------------------------------------------------------------------

@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    body: FeedbackRequest,
    db: Session = Depends(get_db),
):
    correction_saved = False

    feedback_row = Feedback(
        prediction_id=body.prediction_id,
        is_correct=body.is_correct,
        correct_label=body.correct_label,
        image_path=body.image_path,
        comment=body.comment,
    )
    db.add(feedback_row)
    db.commit()
    db.refresh(feedback_row)

    # Copy image to corrections folder when feedback is a correction
    if not body.is_correct and body.correct_label and body.image_path:
        try:
            corrections_dir = Path("ml") / "data" / "corrections" / body.correct_label
            corrections_dir.mkdir(parents=True, exist_ok=True)

            src = Path(body.image_path)
            if src.exists():
                dest = corrections_dir / src.name
                shutil.copy2(src, dest)
                correction_saved = True
        except Exception:
            # Non-fatal: feedback is already saved to DB
            correction_saved = False

    return FeedbackResponse(
        id=feedback_row.id,
        message="Terima kasih atas koreksinya! Data akan digunakan untuk meningkatkan akurasi model.",
        correction_saved=correction_saved,
    )


# ---------------------------------------------------------------------------
# GET /api/feedback/stats
# ---------------------------------------------------------------------------

@router.get("/feedback/stats")
def feedback_stats(db: Session = Depends(get_db)):
    rows = db.query(Feedback).all()
    total = len(rows)

    by_class: dict[str, int] = {}
    for row in rows:
        label = row.correct_label or "__no_correction__"
        by_class[label] = by_class.get(label, 0) + 1

    return {
        "total": total,
        "by_class": by_class,
    }


# ---------------------------------------------------------------------------
# Background retraining helpers
# ---------------------------------------------------------------------------

def _do_retrain() -> None:
    """Run fine-tuning in a background thread and update global status."""
    global _retrain_status

    with _retrain_lock:
        _retrain_status["status"] = "running"
        _retrain_status["message"] = "Retraining sedang berjalan…"

    try:
        from ml.src.finetune import fine_tune_from_corrections  # local import to avoid circular deps

        result = fine_tune_from_corrections()

        with _retrain_lock:
            _retrain_status.update(
                {
                    "status": "done",
                    "message": "Retraining selesai.",
                    "trained_on": result.get("trained_on"),
                    "new_classes": result.get("new_classes"),
                }
            )
    except Exception as exc:  # noqa: BLE001
        with _retrain_lock:
            _retrain_status.update(
                {
                    "status": "error",
                    "message": f"Retraining gagal: {exc}",
                }
            )


# ---------------------------------------------------------------------------
# POST /api/retrain
# ---------------------------------------------------------------------------

@router.post("/retrain", response_model=RetrainStatus)
def trigger_retrain():
    with _retrain_lock:
        if _retrain_status["status"] == "running":
            raise HTTPException(status_code=409, detail="Retraining sedang berjalan")

    thread = threading.Thread(target=_do_retrain, daemon=True)
    thread.start()

    return RetrainStatus(status="started", message="Retraining dimulai di background…")


# ---------------------------------------------------------------------------
# GET /api/retrain/status
# ---------------------------------------------------------------------------

@router.get("/retrain/status", response_model=RetrainStatus)
def retrain_status():
    with _retrain_lock:
        snapshot = dict(_retrain_status)
    return RetrainStatus(**snapshot)
