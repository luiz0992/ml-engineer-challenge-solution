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
import hashlib
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
from api.services.audit_service import AuditService
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
        audit_service: AuditService | None = None,
    ) -> None:
        self.models = model_service
        self.cache = cache_service
        self.settings = settings
        # Optional: the worker and unit tests run without an audit trail, and
        # inference must not depend on one existing.
        self.audit = audit_service or AuditService(None)

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
        user_id: str | None = None,
        user_tier: str | None = None,
        variant: str | None = None,
    ) -> ClassificationResponse:
        """Classify a single image.

        ``variant`` records which A/B arm served the request. It is written to
        the audit trail so the two arms can be compared afterwards; without it
        an experiment produces traffic but no analysable result.
        """
        started = time.perf_counter()

        metadata = validate_image_upload(
            image_bytes,
            max_bytes=self.settings.max_upload_bytes,
            max_pixels=self.settings.max_image_pixels,
        )

        model = self.models.get_classifier(model_version)
        options = {
            "top_k": top_k_results,
            "include_probabilities": include_probabilities,
        }
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()

        cache_key = CacheService.build_key(
            image_bytes,
            model_name=model.name,
            model_version=model.version,
            backend=model.backend.value,
            options=options,
        )

        if use_cache and (cached := await self.cache.get(cache_key)) is not None:
            predictions = [Prediction(**p) for p in cached["predictions"]]
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            self._audit(
                correlation_id,
                model,
                predictions,
                elapsed_ms,
                metadata,
                cached=True,
                user_id=user_id,
                user_tier=user_tier,
                image_sha256=image_sha256,
                variant=variant,
            )
            return ClassificationResponse(
                predictions=predictions,
                inference_time_ms=elapsed_ms,
                correlation_id=correlation_id,
                provenance=_provenance(model),
                cached=True,
            )

        try:
            logits = await self._run_batch(model, [image_bytes])
        except Exception as exc:
            # Failures are audited too: an error rate computed only from
            # successful rows is meaningless, and failures are usually what an
            # investigation is about.
            self._audit(
                correlation_id,
                model,
                [],
                round((time.perf_counter() - started) * 1000, 2),
                metadata,
                cached=False,
                user_id=user_id,
                user_tier=user_tier,
                image_sha256=image_sha256,
                variant=variant,
                status="failure",
                error_code=getattr(exc, "code", type(exc).__name__),
            )
            raise

        predictions = self._decode_classification(
            logits[0], model, top_k_results, include_probabilities
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        response = ClassificationResponse(
            predictions=predictions,
            inference_time_ms=elapsed_ms,
            correlation_id=correlation_id,
            provenance=_provenance(model),
            cached=False,
        )

        if use_cache:
            await self.cache.set(
                cache_key,
                {"predictions": [p.model_dump() for p in predictions]},
            )

        self._audit(
            correlation_id,
            model,
            predictions,
            elapsed_ms,
            metadata,
            cached=False,
            user_id=user_id,
            user_tier=user_tier,
            image_sha256=image_sha256,
            variant=variant,
        )
        return response

    def _audit(
        self,
        correlation_id: str,
        model: LoadedModel,
        predictions: list[Prediction],
        latency_ms: float,
        metadata: Any,
        *,
        cached: bool,
        user_id: str | None,
        user_tier: str | None,
        image_sha256: str | None = None,
        status: str = "success",
        error_code: str | None = None,
        batch_size: int = 1,
        variant: str | None = None,
    ) -> None:
        """Queue an audit record. Never raises and never blocks.

        ``image_sha256`` is a digest of the upload itself, not of its
        dimensions: a fingerprint derived from metadata would collide for every
        image of the same size and be useless for identifying repeat
        submissions or linking a dispute to a specific input.
        """
        top = predictions[0] if predictions else None
        self.audit.record(
            correlation_id=correlation_id,
            user_id=user_id,
            user_tier=user_tier,
            model_name=model.name,
            model_version=model.version,
            backend=model.backend.value,
            task=model.task.value,
            status=status,
            error_code=error_code,
            latency_ms=latency_ms,
            cached=cached,
            batch_size=batch_size,
            image_sha256=image_sha256,
            variant=variant,
            image_bytes=getattr(metadata, "size_bytes", None),
            image_width=getattr(metadata, "width", None),
            image_height=getattr(metadata, "height", None),
            image_format=getattr(metadata, "format", None),
            top_label=top.label if top else None,
            top_class_id=top.class_id if top else None,
            top_probability=top.probability if top else None,
        )

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
