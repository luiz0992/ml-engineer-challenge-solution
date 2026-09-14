"""Shared test fixtures.

Two design decisions shape this file:

**Tests do not depend on the real 87 MB model.** A synthetic ONNX graph is
generated at session scope, which keeps the suite fast and lets it run in CI
without model artifacts. Tests that genuinely need the trained model are marked
``requires_artifacts`` and skip when it is absent, so a missing artifact is a
skip with a reason rather than an error.

**Redis is faked, not mocked.** ``fakeredis`` implements real Redis semantics
including Lua script evaluation, so the rate limiter's sliding-window script is
exercised as written. A ``MagicMock`` would assert only that we called the
methods we call, which proves nothing about whether the algorithm is correct.
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from api.config import AppEnv, InferenceBackend, Settings, UserTier

REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_TEST_CLASSES = 10
TEST_IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at a temporary artifacts directory.

    ``_env_file=None`` prevents a developer's local .env from leaking into the
    test run, which would make results depend on an untracked file.
    """
    return Settings(
        _env_file=None,
        app_env=AppEnv.DEVELOPMENT,
        jwt_secret_key="test-secret-key-not-used-in-production-0123456789abcdef",
        artifacts_dir=tmp_path / "artifacts",
        inference_backend=InferenceBackend.ONNX,
        max_upload_bytes=5 * 1024 * 1024,
        max_image_pixels=10_000_000,
        max_batch_size=8,
        rate_limit_free_rpm=5,
        rate_limit_pro_rpm=20,
        rate_limit_enterprise_rpm=100,
        log_format="console",
    )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def make_image_bytes(
    width: int = 256,
    height: int = 256,
    fmt: str = "JPEG",
    mode: str = "RGB",
    seed: int = 0,
) -> bytes:
    """Encode a deterministic random image.

    Random content rather than a flat colour: a solid image compresses to a few
    bytes and hides size-related bugs, and produces degenerate activations that
    can mask preprocessing errors.
    """
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    if mode != "RGB":
        image = image.convert(mode)

    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image_bytes(fmt="JPEG")


@pytest.fixture
def png_bytes() -> bytes:
    return make_image_bytes(fmt="PNG")


@pytest.fixture
def image_factory():
    """Build images with arbitrary dimensions and formats."""
    return make_image_bytes


# ---------------------------------------------------------------------------
# Synthetic model artifacts
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny ONNX classifier with the same interface as the real one.

    Global-average-pools the input and applies a linear layer, giving a graph
    with the production input name, a dynamic batch axis, and a
    ``(batch, classes)`` output — everything the serving code depends on —
    in a few kilobytes rather than 87 MB.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(
        rng.standard_normal((3, NUM_TEST_CLASSES)).astype(np.float32), name="weight"
    )
    bias = numpy_helper.from_array(np.zeros(NUM_TEST_CLASSES, dtype=np.float32), name="bias")

    nodes = [
        # (N,3,H,W) -> (N,3,1,1)
        helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),  # -> (N,3)
        helper.make_node("MatMul", ["flat", "weight"], ["projected"]),
        helper.make_node("Add", ["projected", "bias"], ["logits"]),
    ]

    graph = helper.make_graph(
        nodes,
        "synthetic_classifier",
        inputs=[
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, ["batch", 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE]
            )
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", NUM_TEST_CLASSES])
        ],
        initializer=[weight, bias],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)

    path = tmp_path_factory.mktemp("synthetic") / "classifier_fp32.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def artifacts_dir(settings: Settings, synthetic_onnx: Path) -> Path:
    """A populated artifacts directory using the synthetic model."""
    import shutil

    root = settings.artifacts_dir
    (root / "onnx").mkdir(parents=True, exist_ok=True)
    shutil.copy2(synthetic_onnx, root / "onnx" / "classifier_fp32.onnx")

    (root / "labels.json").write_text(
        json.dumps(
            {
                "class_names": [f"class_{i}" for i in range(NUM_TEST_CLASSES)],
                "wnids": [f"n{i:08d}" for i in range(NUM_TEST_CLASSES)],
                "image_size": TEST_IMAGE_SIZE,
            }
        )
    )
    (root / "metrics.json").write_text(json.dumps({"final_acc_top1": 0.85}))
    return root


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------
@pytest.fixture
async def fake_redis() -> AsyncIterator[Any]:
    """An in-process Redis implementing real semantics, including Lua."""
    from fakeredis import aioredis

    client = aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def model_service(settings: Settings, artifacts_dir: Path):
    from api.services.model_service import ModelService

    service = ModelService(settings)
    service.load_classifier()
    return service


@pytest.fixture
async def cache_service(fake_redis: Any, settings: Settings):
    from api.services.cache_service import CacheService

    return CacheService(fake_redis, ttl_seconds=settings.cache_ttl_seconds)


@pytest.fixture
def inference_service(model_service, cache_service, settings: Settings):
    from api.services.inference_service import InferenceService

    return InferenceService(model_service, cache_service, settings)


@pytest.fixture
async def rate_limiter(fake_redis: Any, settings: Settings):
    from api.middleware.rate_limit import RateLimiter

    return RateLimiter(fake_redis, settings)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
@pytest.fixture
async def app(
    settings: Settings,
    artifacts_dir: Path,
    fake_redis: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """The FastAPI app with test settings and a fake Redis.

    ``get_settings`` is cached, so the cache is cleared and the override is
    installed before the app is built; otherwise the app would silently pick up
    whichever settings a previous test constructed first.
    """
    from api.config import get_settings
    from api.main import create_app

    get_settings.cache_clear()
    monkeypatch.setattr("api.config.get_settings", lambda: settings)
    monkeypatch.setattr("api.main.get_settings", lambda: settings)

    async def _fake_redis_factory(_: Settings) -> Any:
        return fake_redis

    monkeypatch.setattr("api.main._create_redis", _fake_redis_factory)

    application = create_app(settings)
    application.dependency_overrides[get_settings] = lambda: settings
    return application


@pytest.fixture
async def client(app) -> AsyncIterator[Any]:
    """An HTTP client with the application's lifespan running."""
    from httpx import ASGITransport, AsyncClient

    from api.main import lifespan

    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            yield http_client


@pytest.fixture
async def auth_headers(client: Any) -> dict[str, str]:
    """Headers carrying a pro-tier bearer token."""
    response = await client.post("/api/v1/auth/token", json={"api_key": "dev-key-pro"})
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def principal_factory():
    from api.middleware.auth import Principal

    def _make(user_id: str = "test-user", tier: UserTier = UserTier.PRO) -> Principal:
        return Principal(user_id=user_id, tier=tier)

    return _make


# ---------------------------------------------------------------------------
# Real artifacts (optional)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def real_artifacts_dir() -> Path:
    """The trained model, skipping the test when it has not been published."""
    root = REPO_ROOT / "models" / "artifacts"
    if not (root / "onnx" / "classifier_fp32.onnx").is_file():
        pytest.skip("Trained artifacts not found. Run: uv run python scripts/prepare_artifacts.py")
    return root


@pytest.fixture(scope="session")
def tiny_imagenet_val() -> Path:
    """The validation split, skipping the test when the dataset is absent."""
    root = REPO_ROOT / "data" / "tiny-imagenet-200" / "val"
    if not root.is_dir():
        pytest.skip(
            "Tiny-ImageNet not found. Run: "
            "python scripts/setup/download_datasets.py --dataset tiny_imagenet"
        )
    return root


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_embedder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny ONNX model emitting unit-norm embeddings.

    Mirrors the real embedder's contract -- dynamic batch, L2-normalised
    output -- in a few kilobytes, so similarity tests need no 88 MB artefact.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    dim = 8
    rng = np.random.default_rng(1)
    weight = numpy_helper.from_array(rng.standard_normal((3, dim)).astype(np.float32), name="w")

    nodes = [
        helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "w"], ["raw"]),
        # L2-normalise, exactly as the real export does.
        helper.make_node("ReduceL2", ["raw"], ["norm"], keepdims=1, axes=[1]),
        helper.make_node("Div", ["raw", "norm"], ["embeddings"]),
    ]

    graph = helper.make_graph(
        nodes,
        "synthetic_embedder",
        inputs=[
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, ["batch", 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE]
            )
        ],
        outputs=[helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, ["batch", dim])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    onnx.checker.check_model(model)

    path = tmp_path_factory.mktemp("embedder") / "embedder_fp32.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def similarity_artifacts(artifacts_dir: Path, synthetic_embedder: Path) -> Path:
    """An artifacts directory with an embedder and a small FAISS index."""
    import shutil

    import faiss

    shutil.copy2(synthetic_embedder, artifacts_dir / "onnx" / "embedder_fp32.onnx")

    dim, count = 8, 50
    rng = np.random.default_rng(2)
    vectors = rng.standard_normal((count, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors))
    faiss.write_index(index, str(artifacts_dir / "similarity.index"))

    (artifacts_dir / "similarity_manifest.json").write_text(
        json.dumps(
            {
                "labels": [i % NUM_TEST_CLASSES for i in range(count)],
                "paths": [f"train/class_{i % NUM_TEST_CLASSES}/img_{i}.JPEG" for i in range(count)],
            }
        )
    )
    (artifacts_dir / "similarity_metadata.json").write_text(
        json.dumps(
            {
                "num_vectors": count,
                "embedding_dim": dim,
                "metric": "cosine",
                "source_model": "synthetic",
                "image_size": TEST_IMAGE_SIZE,
                "index_type": "IndexFlatIP",
            }
        )
    )
    return artifacts_dir


@pytest.fixture
def similarity_service(similarity_artifacts: Path, settings: Settings):
    from api.services.model_service import ModelService
    from api.services.similarity_service import SimilarityIndex, SimilarityService

    service = ModelService(settings)
    embedder = service.load_embedder()
    index = SimilarityIndex.load(similarity_artifacts)
    return SimilarityService(service, index, settings, embedder.class_names)
