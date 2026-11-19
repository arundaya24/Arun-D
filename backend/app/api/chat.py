"""Router untuk endpoint Chat AI, Feedback koreksi, dan Retraining active learning."""
from __future__ import annotations

import concurrent.futures
from datetime import datetime
import os
from pathlib import Path
import shutil
import threading
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.database import get_db
from backend.app.models import Feedback, Prediction
from backend.app.schemas.chat import (
    ChatBotResponse,
    FeedbackRequest,
    FeedbackResponse,
    RetrainStatus,
    TopKItem,
)
from backend.app.services.inference import get_predictor
from ml.src.errors import ImageDecodeError

router = APIRouter(prefix="/api", tags=["chat"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
UPLOAD_DIR = Path("ml/data/uploads")
CORRECTIONS_DIR = Path("ml/data/corrections")

_retrain_status: dict = {
    "status": "idle",
    "message": "Belum ada proses retraining yang berjalan.",
    "trained_on": None,
    "new_classes": None,
}
_retrain_lock = threading.Lock()


@router.post("/chat", response_model=ChatBotResponse)
async def chat(
    file: Optional[UploadFile] = File(None),
    message: str = Form(""),
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
):
    # Kasus 1: User mengirimkan gambar
    if file is not None and file.content_type in ALLOWED_TYPES:
        max_bytes = get_settings().max_upload_mb * 1024 * 1024
        data = await file.read()
        if len(data) > max_bytes:
            raise HTTPException(413, f"Ukuran file melebihi {get_settings().max_upload_mb} MB.")
        if not data:
            raise HTTPException(400, "File gambar kosong.")

        try:
            result = predictor.predict_bytes(data, top_k=3)
        except ImageDecodeError as e:
            raise HTTPException(400, str(e)) from e

        # Simpan file sementara agar bisa disimpan jika user memberikan koreksi
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename or "img.jpg").suffix or ".jpg"
        saved_filename = f"{uuid.uuid4().hex[:12]}{ext}"
        saved_path = UPLOAD_DIR / saved_filename
        with open(saved_path, "wb") as f_out:
            f_out.write(data)

        # Simpan ke tabel Prediction
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
        db.refresh(row)

        predicted = result["predicted"]
        conf_pct = round(result["confidence"] * 100, 1)
        is_low_conf = result.get("is_uncertain", result["confidence"] < 0.60)

        # Logika tangani ketidakpastian secara jujur
        if is_low_conf:
            bot_text = (
                f"Saya kurang yakin, namun tebakan terdekat saya adalah **{predicted}** ({conf_pct}%). "
                f"Bisa tolong konfirmasi apakah ini benar gambar {predicted}, atau beri tahu saya objek sebenarnya jika salah?"
            )
        else:
            bot_text = (
                f"Saya cukup yakin ini adalah **{predicted}** (keyakinan: {conf_pct}%). "
                f"Apakah hasil ini sudah sesuai? Anda dapat menekan 'Benar' atau memberikan label koreksi jika salah."
            )

        top_k_items = [TopKItem(label=item["label"], score=item["score"]) for item in result.get("top_k", [])]

        return ChatBotResponse(
            prediction_id=row.id,
            chatbot_message=bot_text,
            predicted=predicted,
            confidence=result["confidence"],
            top_k=top_k_items,
            engine_used=result.get("engine_used", "arun_d_cnn"),
            is_low_confidence=is_low_conf,
            inference_ms=result["inference_ms"],
            model_version=result["model_version"],
        )

    # Kasus 2: Pesan teks saja tanpa gambar
    msg_lower = message.strip().lower()

    if any(q in msg_lower for q in ("ini gambar apa", "apa ini", "ini apa", "gambar apa", "tebak gambar")):
        reply = "Silakan unggah gambar terlebih dahulu dengan tombol ikon kamera/file di samping, lalu saya akan langsung menganalisisnya! 📷"
    elif any(g in msg_lower for g in ("halo", "hai", "hei", "pagi", "siang", "sore", "malam")):
        reply = "Halo! Saya adalah AI asisten pengenal gambar Arun-D. Unggah foto objek apa saja, dan saya akan menebaknya beserta skor keyakinan."
    elif any(t in msg_lower for t in ("terima kasih", "makasih", "thanks")):
        reply = "Sama-sama! Senang bisa membantu. Jika ada gambar lain yang ingin diidentifikasi, silakan kirimkan kapan saja."
    elif not msg_lower:
        reply = "Ada yang bisa saya bantu? Unggah gambar atau tanyakan sesuatu."
    else:
        reply = (
            f"Saya mengerti: \"{message}\". "
            "Untuk mengenali objek, silakan lampirkan gambar dengan menekan ikon kamera di samping kolom chat."
        )

    return ChatBotResponse(
        chatbot_message=reply,
        engine_used="chat_engine",
        is_low_confidence=False,
    )


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    payload: FeedbackRequest,
    db: Session = Depends(get_db),
):
    """Menyimpan feedback pengguna untuk active learning."""
    clean_label = payload.correct_label.strip().lower()
    correction_saved = False
    saved_target_path = None

    # Jika salah, simpan gambar ke dataset koreksi ml/data/corrections/<label>/
    if not payload.is_correct and clean_label:
        dest_dir = CORRECTIONS_DIR / clean_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Cari file dari prediction_id jika ada
        if payload.prediction_id:
            pred = db.get(Prediction, payload.prediction_id)
            if pred and UPLOAD_DIR.exists():
                # Cari berkas terbaru di UPLOAD_DIR
                upload_files = sorted(UPLOAD_DIR.glob("*"), key=os.path.getmtime, reverse=True)
                if upload_files:
                    src_file = upload_files[0]
                    target_file = dest_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{src_file.name}"
                    try:
                        shutil.copy2(src_file, target_file)
                        saved_target_path = str(target_file)
                        correction_saved = True
                    except Exception:
                        pass

    fb_row = Feedback(
        prediction_id=payload.prediction_id,
        original_label=payload.original_label[:100],
        correct_label=clean_label[:100] if clean_label else payload.original_label[:100],
        image_path=saved_target_path,
        is_correct=payload.is_correct,
    )
    db.add(fb_row)
    db.commit()
    db.refresh(fb_row)

    msg = (
        "Terima kasih atas konfirmasinya! 👍"
        if payload.is_correct
        else f"Terima kasih atas koreksinya! Label '{clean_label}' telah disimpan untuk melatih ulang AI agar lebih pintar."
    )

    return FeedbackResponse(
        id=fb_row.id,
        message=msg,
        correction_saved=correction_saved,
    )


@router.get("/feedback/stats")
def feedback_stats(db: Session = Depends(get_db)):
    """Mengambil ringkasan data koreksi yang masuk."""
    total = db.scalar(select(func.count(Feedback.id))) or 0
    correct_count = db.scalar(select(func.count(Feedback.id)).where(Feedback.is_correct == True)) or 0  # noqa: E712
    wrong_count = total - correct_count

    by_label = db.execute(
        select(Feedback.correct_label, func.count(Feedback.id))
        .where(Feedback.is_correct == False)  # noqa: E712
        .group_by(Feedback.correct_label)
        .order_by(func.count(Feedback.id).desc())
    ).all()

    return {
        "total_feedback": total,
        "correct_predictions": correct_count,
        "corrections_received": wrong_count,
        "top_corrections": [{"label": row[0], "count": row[1]} for row in by_label[:10]],
    }


def _run_retraining_task():
    global _retrain_status
    with _retrain_lock:
        _retrain_status["status"] = "running"
        _retrain_status["message"] = "Sedang melatih ulang model dengan data koreksi..."

    try:
        from ml.src.finetune import fine_tune_from_corrections
        res = fine_tune_from_corrections(
            corrections_dir="ml/data/corrections",
            artifacts_dir="ml/artifacts",
            epochs=5,
        )
        with _retrain_lock:
            _retrain_status["status"] = "success"
            _retrain_status["message"] = f"Retraining selesai! Dilatih pada {res['trained_on']} data. Versi baru: {res.get('model_version')}"
            _retrain_status["trained_on"] = res["trained_on"]
            _retrain_status["new_classes"] = res["new_classes"]
    except Exception as e:
        with _retrain_lock:
            _retrain_status["status"] = "error"
            _retrain_status["message"] = f"Gagal retraining: {str(e)}"


@router.post("/retrain", response_model=RetrainStatus)
def trigger_retraining():
    """Memulai proses fine-tuning model dari data koreksi user."""
    global _retrain_status
    with _retrain_lock:
        if _retrain_status["status"] == "running":
            raise HTTPException(409, "Proses retraining sedang berjalan di background.")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(_run_retraining_task)

    return RetrainStatus(
        status="started",
        message="Proses retraining telah dimulai di latar belakang.",
    )


@router.get("/retrain/status", response_model=RetrainStatus)
def get_retrain_status():
    """Mengecek status proses retraining."""
    with _retrain_lock:
        return RetrainStatus(
            status=_retrain_status["status"],
            message=_retrain_status["message"],
            trained_on=_retrain_status.get("trained_on"),
            new_classes=_retrain_status.get("new_classes"),
        )
