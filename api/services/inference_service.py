"""Inference orchestration.

Sits between the routers and the model runtime, owning the sequence that every
prediction follows: validate, check cache, preprocess, run, post-process,
store. Keeping that sequence in one place means the HTTP layer contains no ML
logic and the model layer contains no HTTP concerns, so either can be tested
without the other.

The forward pass runs in a thread pool. ONNX Runtime's ``Run`` is a blocking
call that releases the GIL; executing it directly in the event loop would stall
every other in-flight request for its duration, which at batch 32 is tens of
milliseconds — enough to destroy tail latency under concurrency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from api.config import Settings
from api.exceptions import BatchTooLargeError
from api.logging_config import get_logger
from api.models.responses import (
    ClassificationResponse,
    ModelProvenance,
    Prediction,
)
from api.services.cache_service import CacheService
from api.services.model_service import LoadedModel, ModelService
from api.utils.image_processing import (
    PreprocessConfig,
    preprocess_image,
    softmax,
    stack_batch,
    top_k,
)
from api.utils.validators import validate_image_upload

logger = get_logger(__name__)


class InferenceService:
    """Runs the end-to-end prediction pipeline."""

    def __init__(
        self,
        model_service: ModelService,
        cache_service: CacheService,
        settings: Settings,
    ) -> None:
        self.models = model_service
        self.cache = cache_service
        self.settings = settings

    # --- Classification ---------------------------------------------------
    async def classify(
        self,
        image_bytes: bytes,
        *,
        correlation_id: str,
        top_k_results: int = 5,
        model_version: str | None = None,
        include_probabilities: bool = True,
        use_cache: bool = True,
    ) -> ClassificationResponse:
        """Classify a single image."""
        started = time.perf_counter()

        validate_image_upload(
            image_bytes,
            max_bytes=self.settings.max_upload_bytes,
            max_pixels=self.settings.max_image_pixels,
        )

        model = self.models.get_classifier(model_version)
        options = {
            "top_k": top_k_results,
            "include_probabilities": include_probabilities,
        }

        cache_key = CacheService.build_key(
            image_bytes,
            model_name=model.name,
            model_version=model.version,
            backend=model.backend.value,
            options=options,
        )

        if use_cache and (cached := await self.cache.get(cache_key)) is not None:
            return ClassificationResponse(
                predictions=[Prediction(**p) for p in cached["predictions"]],
                inference_time_ms=round((time.perf_counter() - started) * 1000, 2),
                correlation_id=correlation_id,
                provenance=_provenance(model),
                cached=True,
            )

        logits = await self._run_batch(model, [image_bytes])
        predictions = self._decode_classification(
            logits[0], model, top_k_results, include_probabilities
        )

        response = ClassificationResponse(
            predictions=predictions,
            inference_time_ms=round((time.perf_counter() - started) * 1000, 2),
            correlation_id=correlation_id,
            provenance=_provenance(model),
            cached=False,
        )

        if use_cache:
            await self.cache.set(
                cache_key,
                {"predictions": [p.model_dump() for p in predictions]},
            )

        return response

    async def classify_many(
        self,
        images: list[bytes],
        *,
        top_k_results: int = 5,
        model_version: str | None = None,
        include_probabilities: bool = True,
    ) -> list[list[Prediction]]:
        """Classify several images in one forward pass.

        Used by the batch worker. Batching is what makes throughput scale: at
        batch 32 the model achieves roughly 4,800 images/s against 760 at batch
        1, because a single image cannot saturate the GPU.
        """
        if not images:
            return []
        if len(images) > self.settings.max_batch_size:
            raise BatchTooLargeError(
                f"Batch of {len(images)} exceeds the limit of {self.settings.max_batch_size}.",
                details={"submitted": len(images), "max": self.settings.max_batch_size},
            )

        for image in images:
            validate_image_upload(
                image,
                max_bytes=self.settings.max_upload_bytes,
                max_pixels=self.settings.max_image_pixels,
            )

        model = self.models.get_classifier(model_version)
        logits = await self._run_batch(model, images)

        return [
            self._decode_classification(row, model, top_k_results, include_probabilities)
            for row in logits
        ]

    # --- Internals --------------------------------------------------------
    async def _run_batch(self, model: LoadedModel, images: list[bytes]) -> np.ndarray:
        """Preprocess and run a batch off the event loop."""
        config = PreprocessConfig(image_size=model.image_size)

        def _work() -> np.ndarray:
            batch = stack_batch([preprocess_image(image, config) for image in images])
            return self.models.run(model, batch)

        # to_thread keeps both the CPU-bound preprocessing and the blocking
        # Run() call off the event loop.
        return await asyncio.to_thread(_work)

    @staticmethod
    def _decode_classification(
        logits: np.ndarray,
        model: LoadedModel,
        top_k_results: int,
        include_probabilities: bool,
    ) -> list[Prediction]:
        """Turn raw logits into ranked, labelled predictions."""
        probabilities = softmax(logits)
        scores, indices = top_k(probabilities, top_k_results)

        predictions: list[Prediction] = []
        for score, index in zip(scores, indices, strict=True):
            class_id = int(index)
            predictions.append(
                Prediction(
                    label=model.class_names[class_id],
                    class_id=class_id,
                    wnid=model.wnids[class_id] if class_id < len(model.wnids) else None,
                    probability=round(float(score), 6) if include_probabilities else None,
                )
            )
        return predictions


def _provenance(model: LoadedModel) -> ModelProvenance:
    return ModelProvenance(
        model_name=model.name,
        model_version=model.version,
        backend=model.backend.value,
    )


def summarise_predictions(predictions: list[Prediction]) -> dict[str, Any]:
    """Compact representation used in batch results and logs."""
    return {
        "predictions": [
            {"label": p.label, "class_id": p.class_id, "probability": p.probability}
            for p in predictions
        ]
    }
