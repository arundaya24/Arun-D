"""Tes backend tanpa PyTorch: Predictor diganti objek tiruan.
Jalankan dari root: python -m pytest backend/tests
"""
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_arun_d.db"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.core.database import Base, engine  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.services.inference import get_predictor  # noqa: E402
from ml.src.errors import ImageDecodeError  # noqa: E402


class FakePredictor:
    def predict_bytes(self, data, top_k=3):
        if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
            raise ImageDecodeError("File bukan gambar yang valid.")
        return {
            "predicted": "cat", "confidence": 0.9,
            "top_k": [{"label": "cat", "score": 0.9}, {"label": "dog", "score": 0.1}],
            "inference_ms": 5.0, "model_version": "test-v0",
        }


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _png() -> bytes:
    ok, buf = cv2.imencode(".png", np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_predict_history_stats_flow(client):
    r = client.post("/api/predict", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted"] == "cat" and body["confidence"] == 0.9

    h = client.get("/api/history").json()
    assert h["total"] == 1 and h["items"][0]["filename"] == "a.png"

    s = client.get("/api/stats").json()
    assert s["total_predictions"] == 1
    assert s["by_class"][0] == {"label": "cat", "count": 1}
    assert len(s["daily"]) == 1


def test_rejects_wrong_type(client):
    r = client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/api/predict", files={"file": ("a.png", b"bukan gambar", "image/png")})
    assert r.status_code == 400


def test_delete_history(client):
    pid = client.post("/api/predict", files={"file": ("a.png", _png(), "image/png")}).json()["id"]
    assert client.delete(f"/api/history/{pid}").status_code == 204
    assert client.delete(f"/api/history/{pid}").status_code == 404


def test_model_info_404_when_untrained(client, monkeypatch, tmp_path):
    from backend.app.core.config import get_settings
    monkeypatch.setattr(get_settings(), "model_dir", str(tmp_path))
    assert client.get("/api/model/info").status_code == 404
