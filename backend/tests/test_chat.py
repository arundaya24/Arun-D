import os
os.environ["DATABASE_URL"] = "sqlite:///./test_arun_d.db"

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.database import Base, engine
from backend.app.main import app
from backend.app.services.inference import get_predictor


class FakeChatPredictor:
    def __init__(self, confidence=0.85, is_human=False, label="cat"):
        self.confidence = confidence
        self.is_human = is_human
        self.label = label

    def predict_bytes(self, data, top_k=3):
        return {
            "predicted": self.label,
            "confidence": self.confidence,
            "top_k": [{"label": self.label, "score": self.confidence}, {"label": "dog", "score": round(1 - self.confidence, 4)}],
            "inference_ms": 12.5,
            "model_version": "test-v1",
            "engine_used": "fake_engine",
            "is_uncertain": self.confidence < 0.60,
            "is_human": self.is_human,
        }


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_predictor] = lambda: FakeChatPredictor(confidence=0.85)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _dummy_image() -> bytes:
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_chat_text_only_prompt(client):
    r = client.post("/api/chat", data={"message": "ini gambar apa?"})
    assert r.status_code == 200
    body = r.json()
    assert "Unggah berkas citra" in body["chatbot_message"]


def test_chat_with_image_high_confidence(client):
    r = client.post(
        "/api/chat",
        files={"file": ("test.png", _dummy_image(), "image/png")},
        data={"message": "ini gambar apa"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["predicted"] == "cat"
    assert body["confidence"] == 0.85
    assert not body["is_low_confidence"]
    assert "Objek teridentifikasi" in body["chatbot_message"]


def test_chat_with_image_low_confidence(client):
    app.dependency_overrides[get_predictor] = lambda: FakeChatPredictor(confidence=0.45)
    r = client.post(
        "/api/chat",
        files={"file": ("test.png", _dummy_image(), "image/png")},
        data={"message": "coba tebak"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_low_confidence"] is True
    assert "Tingkat keyakinan model" in body["chatbot_message"]


def test_chat_human_detection(client):
    app.dependency_overrides[get_predictor] = lambda: FakeChatPredictor(confidence=0.91, is_human=True, label="human")
    r = client.post(
        "/api/chat",
        files={"file": ("person.png", _dummy_image(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_human"] is True
    assert "manusia" in body["chatbot_message"].lower()


def test_feedback_and_stats(client):
    # 1. Jalankan prediksi chat
    pred_res = client.post(
        "/api/chat",
        files={"file": ("test.png", _dummy_image(), "image/png")},
    ).json()
    pred_id = pred_res["prediction_id"]

    # 2. Kirim feedback salah + koreksi label
    fb_res = client.post(
        "/api/feedback",
        json={
            "prediction_id": pred_id,
            "original_label": "cat",
            "correct_label": "kucing anggora",
            "is_correct": False,
        },
    )
    assert fb_res.status_code == 200
    assert "Koreksi label" in fb_res.json()["message"]

    # 3. Cek statistik feedback
    stats_res = client.get("/api/feedback/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_feedback"] == 1
    assert stats["corrections_received"] == 1
    assert stats["top_corrections"][0]["label"] == "kucing anggora"


def test_retrain_status(client):
    r = client.get("/api/retrain/status")
    assert r.status_code == 200
    assert r.json()["status"] in ("idle", "running", "success")
