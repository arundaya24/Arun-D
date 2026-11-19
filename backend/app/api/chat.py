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
    # Kasus 1: User mengirimkan berkas gambar
    has_image = False
    if file is not None and file.filename:
        ctype = file.content_type or ""
        fname = file.filename.lower()
        if ctype.startswith("image/") or any(fname.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".tiff")):
            has_image = True

    if has_image and file is not None:
        max_bytes = get_settings().max_upload_mb * 1024 * 1024
        data = await file.read()
        if len(data) > max_bytes:
            raise HTTPException(413, f"Ukuran file melebihi {get_settings().max_upload_mb} MB.")
        if not data:
            raise HTTPException(400, "Berkas gambar kosong.")

        try:
            result = predictor.predict_bytes(data, top_k=3)
        except ImageDecodeError as e:
            raise HTTPException(400, str(e)) from e

        # Simpan file sementara agar bisa dipakai jika ada koreksi
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
        is_human = result.get("is_human", False) or predicted == "human"

        if is_human:
            bot_text = (
                f"Subjek citra terdeteksi sebagai manusia (keyakinan: {conf_pct}%). "
                "Konfirmasikan apakah deteksi ini sesuai atau masukkan label spesifik."
            )
        elif is_low_conf:
            bot_text = (
                f"Tingkat keyakinan model berada di bawah batas konfirmasi (prediksi: {predicted}, {conf_pct}%). "
                f"Konfirmasikan apakah objek ini adalah {predicted} atau berikan koreksi label."
            )
        else:
            bot_text = (
                f"Objek teridentifikasi sebagai {predicted} dengan tingkat keyakinan {conf_pct}%. "
                "Silakan konfirmasi kebenaran hasil atau masukkan label baru jika salah."
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
            is_human=is_human,
            inference_ms=result["inference_ms"],
            model_version=result["model_version"],
        )

    # Kasus 2: Pesan teks saja tanpa gambar
    msg_lower = message.strip().lower()

    if any(q in msg_lower for q in ("ini gambar apa", "apa ini", "ini apa", "gambar apa", "tebak gambar")):
        reply = "Unggah berkas citra terlebih dahulu melalui tombol input di samping kolom teks untuk menjalankan inferensi."
    elif any(g in msg_lower for g in ("halo", "hai", "hei", "pagi", "siang", "sore", "malam")):
        reply = "Sistem klasifikasi citra siap menerima input. Unggah berkas citra untuk melihat telemetri inferensi."
    elif any(t in msg_lower for t in ("terima kasih", "makasih", "thanks")):
        reply = "Layanan aktif. Silakan unggah berkas berikutnya kapan saja."
    elif not msg_lower:
        reply = "Sistem menunggu input. Lampirkan berkas citra atau masukkan instruksi."
    else:
        reply = (
            f"Input diterima: \"{message}\". "
            "Untuk menjalankan klasifikasi, lampirkan berkas citra pada formulir."
        )

    return ChatBotResponse(
        chatbot_message=reply,
        engine_used="chat_engine",
        is_low_confidence=False,
        is_human=False,
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

    if not payload.is_correct and clean_label:
        dest_dir = CORRECTIONS_DIR / clean_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        if payload.prediction_id:
            pred = db.get(Prediction, payload.prediction_id)
            if pred and UPLOAD_DIR.exists():
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
        "Konfirmasi kebenaran label berhasil dicatat ke basis data."
        if payload.is_correct
        else f"Koreksi label '{clean_label}' tersimpan ke dataset retraining aktif."
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
        _retrain_status["message"] = "Fine-tuning model dengan data koreksi sedang diproses."

    try:
        from ml.src.finetune import fine_tune_from_corrections
        res = fine_tune_from_corrections(
            corrections_dir="ml/data/corrections",
            artifacts_dir="ml/artifacts",
            epochs=5,
        )
        try:
            from backend.app.services.inference import reload_predictor
            reload_predictor()
        except Exception:
            pass
        with _retrain_lock:
            _retrain_status["status"] = "success"
            _retrain_status["message"] = f"Retraining selesai. Total sampel: {res['trained_on']}. Versi: {res.get('model_version')}"
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
            raise HTTPException(409, "Proses retraining sedang berjalan.")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(_run_retraining_task)

    return RetrainStatus(
        status="started",
        message="Proses retraining telah diinisialisasi pada thread terpisah.",
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
