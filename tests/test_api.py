import io
import json
import numpy as np
import cv2
import pytest

from app.config import Config
from app.face_engine import DetectedFace, face_engine
from app.main import create_app


def _make_test_image_bytes(noisy: bool = True) -> bytes:
    if noisy:
        rng = np.random.default_rng(42)
        img = rng.integers(60, 200, size=(300, 300, 3), dtype=np.uint8)
    else:
        img = np.full((300, 300, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(monkeypatch):
    face_engine.ready = True
    app = create_app(load_model=False)
    app.config["TESTING"] = True
    return app.test_client()


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["model"]["name"] == Config.MODEL_NAME


def test_requires_internal_token(client):
    resp = client.post("/v1/detect")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "UNAUTHORIZED"


def test_embed_no_face(client, monkeypatch):
    monkeypatch.setattr(face_engine, "analyze", lambda _img: [])
    resp = client.post(
        "/v1/embed",
        headers={"X-Internal-Token": Config.INTERNAL_TOKEN},
        data={"image": (io.BytesIO(_make_test_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 0
    assert data["error"]["code"] == "NO_FACE"


def test_embed_multiple_faces(client, monkeypatch):
    monkeypatch.setattr(
        face_engine,
        "analyze",
        lambda _img: [
            DetectedFace(bbox=[10, 10, 100, 100], score=0.99, embedding=[0.1, 0.2]),
            DetectedFace(bbox=[150, 10, 100, 100], score=0.98, embedding=[0.2, 0.1]),
        ],
    )
    resp = client.post(
        "/v1/embed",
        headers={"X-Internal-Token": Config.INTERNAL_TOKEN},
        data={"image": (io.BytesIO(_make_test_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 2
    assert data["error"]["code"] == "MULTIPLE_FACES"


def test_embed_success(client, monkeypatch):
    vec = [1.0] + [0.0] * 511
    monkeypatch.setattr(
        face_engine,
        "analyze",
        lambda _img: [DetectedFace(bbox=[50, 50, 150, 150], score=0.99, embedding=vec)],
    )
    resp = client.post(
        "/v1/embed",
        headers={"X-Internal-Token": Config.INTERNAL_TOKEN},
        data={"image": (io.BytesIO(_make_test_image_bytes(noisy=True)), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 1
    assert data["quality"]["passed"] is True
    assert data["dim"] == 512
    assert len(data["embedding"]) == 512


def test_verify_success(client, monkeypatch):
    probe = [1.0, 0.0, 0.0]
    ref_match = [1.0, 0.0, 0.0]
    ref_other = [0.0, 1.0, 0.0]

    monkeypatch.setattr(
        face_engine,
        "analyze",
        lambda _img: [DetectedFace(bbox=[50, 50, 150, 150], score=0.99, embedding=probe)],
    )
    resp = client.post(
        "/v1/verify",
        headers={"X-Internal-Token": Config.INTERNAL_TOKEN},
        data={
            "image": (io.BytesIO(_make_test_image_bytes(noisy=True)), "checkin.jpg"),
            "references": json.dumps([ref_match, ref_other]),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 1
    assert data["similarities"] == [1.0, 0.0]
    assert data["similarity_max"] == 1.0
