import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as api_module
from app.main import app
from app.model_loader import model_state


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


class FakeModel:
    def predict(self, _input_tensor, verbose=0):
        return np.array([[0.05, 0.10, 0.55, 0.20, 0.10]], dtype=np.float32)


@pytest.fixture
def client(monkeypatch):
    def load_test_model():
        with (MODEL_DIR / "config.json").open(encoding="utf-8") as config_file:
            model_state.config = json.load(config_file)
        model_state.model = FakeModel()

    monkeypatch.setattr(model_state, "load", load_test_model)
    monkeypatch.setattr(
        api_module,
        "make_gradcam",
        lambda _model, _input_tensor: np.full((14, 14), 0.5, dtype=np.float32),
    )
    with TestClient(app) as test_client:
        yield test_client
    model_state.model = None
    model_state.config = None


def make_fundus_png_bytes():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 110, (55, 85, 145), thickness=-1)
    for endpoint in [(40, 100), (215, 70), (205, 190), (55, 205), (130, 25)]:
        cv2.line(image, (128, 128), endpoint, (95, 120, 185), thickness=2)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_health_reports_loaded_model_when_model_file_exists(client):
    assert (MODEL_DIR / "dr_model.keras").is_file()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_contract_and_gradcam_png(client):
    response = client.post(
        "/predict",
        files={"file": ("fundus.png", io.BytesIO(make_fundus_png_bytes()), "image/png")},
    )

    assert response.status_code == 200
    result = response.json()
    assert set(result) == {
        "grade",
        "label",
        "confidence",
        "probabilities",
        "low_confidence",
        "threshold",
        "gradcam_base64",
    }
    assert isinstance(result["grade"], int)
    assert isinstance(result["label"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["probabilities"], list)
    assert all(isinstance(value, float) for value in result["probabilities"])
    assert isinstance(result["low_confidence"], bool)
    assert isinstance(result["threshold"], float)
    assert isinstance(result["gradcam_base64"], str)
    assert len(result["probabilities"]) == 5
    assert sum(result["probabilities"]) == pytest.approx(1.0, abs=1e-5)
    assert result["grade"] == int(np.argmax(result["probabilities"]))
    assert result["confidence"] == pytest.approx(max(result["probabilities"]))
    assert result["low_confidence"] == (result["confidence"] < result["threshold"])

    gradcam_bytes = base64.b64decode(result["gradcam_base64"], validate=True)
    gradcam = cv2.imdecode(np.frombuffer(gradcam_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert gradcam is not None
    assert gradcam.shape == (224, 224, 3)


@pytest.mark.parametrize(
    ("contents", "filename", "content_type", "expected_status"),
    [
        (b"not an image", "renamed.jpg", "image/jpeg", 400),
        (b"", "empty.jpg", "image/jpeg", 400),
        (b"x" * (10 * 1024 * 1024 + 1), "large.jpg", "image/jpeg", 413),
    ],
    ids=["text-file", "empty-file", "oversize-file"],
)
def test_predict_rejects_invalid_files(
    client, contents, filename, content_type, expected_status
):
    response = client.post(
        "/predict",
        files={"file": (filename, io.BytesIO(contents), content_type)},
    )

    assert response.status_code == expected_status
    assert "detail" in response.json()


def test_predict_requires_file_field(client):
    response = client.post("/predict")

    assert response.status_code == 422
