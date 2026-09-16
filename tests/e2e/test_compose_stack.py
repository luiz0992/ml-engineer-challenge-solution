"""End-to-end checks against a running Compose stack.

Skipped unless ``E2E_BASE_URL`` is set, so the default unit/integration run
stays fast and artefact-free. CI's docker job starts Compose and exports
the variable.

Self-contained: nothing here imports ``api`` or ``tests.conftest``, so CI can
run it in a slim image holding only httpx, pytest, pillow, and numpy. That
only holds if this directory is mounted *alone* — pytest loads every
conftest.py between its rootdir and the test file, and the suite's top-level
conftest imports the full serving stack.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.e2e

BASE_URL = os.environ.get("E2E_BASE_URL")


def _image_bytes() -> bytes:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def base_url() -> str:
    if not BASE_URL:
        pytest.skip("E2E_BASE_URL is not set; Compose stack is not under test")
    return BASE_URL.rstrip("/")


@pytest.fixture(scope="module")
def client(base_url: str):
    import httpx

    with httpx.Client(base_url=base_url, timeout=30.0) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def token(client) -> str:
    response = client.post("/api/v1/auth/token", json={"api_key": "dev-key-pro"})
    response.raise_for_status()
    return response.json()["access_token"]


class TestComposeStack:
    def test_liveness(self, client) -> None:
        response = client.get("/health/live")
        assert response.status_code == 200

    def test_readiness_reports_a_loaded_model(self, client) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"healthy", "degraded"}
        names = {c["name"] for c in body["components"]}
        assert "models" in names

    def test_classify_through_the_gateway(self, client, token: str) -> None:
        response = client.post(
            "/api/v1/classify",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("image.jpg", _image_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["predictions"]
        assert body["provenance"]["model_name"] == "tiny-imagenet-classifier"
        assert body["inference_time_ms"] < 1000

    def test_models_lists_the_classifier(self, client) -> None:
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        names = {m["name"] for m in response.json()["models"]}
        assert "tiny-imagenet-classifier" in names

    def test_detect_through_the_gateway(self, client, token: str) -> None:
        response = client.post(
            "/api/v1/detect",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("image.jpg", _image_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provenance"]["model_name"] == "rtdetr-coco-detector"
        assert body["image_width"] == 256
        assert body["image_height"] == 256
        for detection in body["detections"]:
            box = detection["box"]
            assert 0.0 <= box["x_min"] <= box["x_max"] <= 256.0
            assert 0.0 <= box["y_min"] <= box["y_max"] <= 256.0

    def test_similar_through_the_gateway(self, client, token: str) -> None:
        response = client.post(
            "/api/v1/similar?top_k=5",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("image.jpg", _image_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provenance"]["model_name"] == "tiny-imagenet-embedder"
        assert body["index_size"] > 0
        assert 1 <= len(body["results"]) <= 5
