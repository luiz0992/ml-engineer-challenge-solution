"""Tests for inference orchestration.

Covers the batch path and the degraded-service reporting that the single-image
integration tests do not reach.
"""

from __future__ import annotations

from typing import Any

import pytest

from api.config import InferenceBackend, Settings
from api.exceptions import BatchTooLargeError, InvalidImageError
from tests.conftest import make_image_bytes

pytestmark = pytest.mark.unit


class TestBatchClassification:
    async def test_classifies_several_images_in_one_pass(self, inference_service: Any) -> None:
        images = [make_image_bytes(seed=i) for i in range(4)]

        results = await inference_service.classify_many(images, top_k_results=3)

        assert len(results) == 4
        assert all(len(predictions) == 3 for predictions in results)

    async def test_results_are_ordered_to_match_inputs(self, inference_service: Any) -> None:
        """Result i must correspond to input i.

        A reordering would attach every prediction to the wrong image, which no
        shape or type check would catch.
        """
        images = [make_image_bytes(seed=i) for i in range(5)]

        batched = await inference_service.classify_many(images, top_k_results=1)
        individually = [
            (await inference_service.classify_many([image], top_k_results=1))[0] for image in images
        ]

        assert [p[0].class_id for p in batched] == [p[0].class_id for p in individually]

    async def test_empty_batch_returns_empty(self, inference_service: Any) -> None:
        assert await inference_service.classify_many([]) == []

    async def test_rejects_a_batch_over_the_limit(
        self, inference_service: Any, settings: Settings
    ) -> None:
        images = [make_image_bytes(seed=i) for i in range(settings.max_batch_size + 1)]

        with pytest.raises(BatchTooLargeError) as exc_info:
            await inference_service.classify_many(images)

        assert exc_info.value.details["max"] == settings.max_batch_size

    async def test_one_invalid_image_rejects_the_whole_batch(self, inference_service: Any) -> None:
        """Validation happens before inference, so the batch fails as a unit.

        This is intentional for the synchronous path: the caller can fix their
        input and retry. The asynchronous batch endpoint takes the opposite
        approach and reports per-item failures, because there a single bad URL
        must not discard hundreds of good results.
        """
        images = [make_image_bytes(), b"not an image", make_image_bytes()]

        with pytest.raises(InvalidImageError):
            await inference_service.classify_many(images)


class TestPredictionDecoding:
    async def test_probabilities_form_a_distribution(self, inference_service: Any) -> None:
        response = await inference_service.classify(
            make_image_bytes(), correlation_id="c", top_k_results=10, use_cache=False
        )
        total = sum(p.probability for p in response.predictions)

        # All ten classes of the synthetic model, so the total is the full mass.
        assert total == pytest.approx(1.0, abs=1e-5)

    async def test_predictions_are_ranked(self, inference_service: Any) -> None:
        response = await inference_service.classify(
            make_image_bytes(), correlation_id="c", top_k_results=5, use_cache=False
        )
        probabilities = [p.probability for p in response.predictions]

        assert probabilities == sorted(probabilities, reverse=True)

    async def test_top_k_is_clamped_to_available_classes(self, inference_service: Any) -> None:
        """Asking for more classes than exist returns all of them, not an error."""
        response = await inference_service.classify(
            make_image_bytes(), correlation_id="c", top_k_results=100, use_cache=False
        )
        assert len(response.predictions) == 10

    async def test_reports_the_backend_actually_used(self, inference_service: Any) -> None:
        response = await inference_service.classify(
            make_image_bytes(), correlation_id="c", use_cache=False
        )
        assert response.provenance.backend in {b.value for b in InferenceBackend}


class TestHealthReporting:
    async def test_degraded_backend_is_surfaced(
        self, settings: Settings, artifacts_dir: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fallback must be visible, not silent.

        A service quietly running on a slower backend looks healthy while
        missing its latency budget; the degradation has to reach /health.
        """
        from pathlib import Path

        from api.services.model_service import ModelService

        settings = settings.model_copy(update={"inference_backend": InferenceBackend.TENSORRT})
        original = ModelService._create_session

        def _fail_tensorrt(self: ModelService, artifact: Path, backend: InferenceBackend):
            if backend is InferenceBackend.TENSORRT:
                raise RuntimeError("engine build failed")
            return original(self, artifact, backend)

        monkeypatch.setattr(ModelService, "_create_session", _fail_tensorrt)

        service = ModelService(settings)
        model = service.load_classifier()

        assert model.degraded_from is InferenceBackend.TENSORRT

        from api.routers.health import health
        from api.services.cache_service import CacheService

        response = await health(models=service, cache=CacheService(None))

        # Degraded, not unhealthy: the service is serving and must stay in the
        # load balancer.
        assert response.status == "degraded"
        assert any(c.name == "inference_backend" for c in response.components)

    async def test_unhealthy_when_no_model_is_loaded(self, settings: Settings) -> None:
        from api.routers.health import health
        from api.services.cache_service import CacheService
        from api.services.model_service import ModelService

        response = await health(models=ModelService(settings), cache=CacheService(None))

        assert response.status == "unhealthy"
        assert any(c.name == "models" and not c.healthy for c in response.components)

    async def test_readiness_is_false_before_a_model_loads(self, settings: Settings) -> None:
        from api.routers.health import readiness
        from api.services.model_service import ModelService

        body = await readiness(models=ModelService(settings), response=None)  # type: ignore[arg-type]
        assert body["ready"] is False
