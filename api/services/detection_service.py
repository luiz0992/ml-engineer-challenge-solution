"""Object detection inference.

Post-processing is where a detector is most often silently wrong, so the two
decisions that matter are made explicit here:

**Per-class sigmoid, not softmax.** RT-DETR scores each of its 300 queries
against every class independently, because a region may legitimately match more
than one label. Applying softmax across classes would force the scores to sum
to one and systematically suppress confident multi-label detections. The
sigmoid is baked into the exported graph so a reimplementation cannot get it
wrong.

**No non-maximum suppression.** RT-DETR predicts a fixed set of queries with a
one-to-one assignment loss, so duplicate boxes are suppressed during training
rather than at inference. Adding NMS here would remove legitimate detections of
genuinely overlapping objects.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from api.config import Settings
from api.logging_config import get_logger
from api.models.responses import (
    BoundingBox,
    Detection,
    DetectionResponse,
    ModelProvenance,
)
from api.services.model_service import LoadedModel, ModelService
from api.utils.image_processing import boxes_to_absolute, preprocess_for_detection
from api.utils.validators import validate_image_upload

logger = get_logger(__name__)


class DetectionService:
    """Runs object detection over uploaded images."""

    def __init__(self, model_service: ModelService, settings: Settings) -> None:
        self.models = model_service
        self.settings = settings

    async def detect(
        self,
        image_bytes: bytes,
        *,
        correlation_id: str,
        confidence_threshold: float = 0.5,
        max_detections: int = 100,
        model_version: str | None = None,
    ) -> DetectionResponse:
        """Detect objects and return boxes in the uploaded image's pixel space."""
        started = time.perf_counter()

        validate_image_upload(
            image_bytes,
            max_bytes=self.settings.max_upload_bytes,
            max_pixels=self.settings.max_image_pixels,
        )

        model = self.models.get_detector(model_version)

        def _work() -> tuple[np.ndarray, np.ndarray, int, int]:
            array, width, height = preprocess_for_detection(image_bytes, model.image_size)
            scores, boxes = self.models.run_multi(model, array[None, ...])
            return scores[0], boxes[0], width, height

        # Off the event loop: ONNX Runtime's Run() blocks, and detection is
        # several times heavier than classification.
        scores, boxes, width, height = await asyncio.to_thread(_work)

        detections = self._decode(
            scores, boxes, model, width, height, confidence_threshold, max_detections
        )

        return DetectionResponse(
            detections=detections,
            image_width=width,
            image_height=height,
            inference_time_ms=round((time.perf_counter() - started) * 1000, 2),
            correlation_id=correlation_id,
            provenance=ModelProvenance(
                model_name=model.name,
                model_version=model.version,
                backend=model.backend.value,
            ),
            cached=False,
        )

    @staticmethod
    def _decode(
        scores: np.ndarray,
        boxes: np.ndarray,
        model: LoadedModel,
        width: int,
        height: int,
        threshold: float,
        max_detections: int,
    ) -> list[Detection]:
        """Turn raw query outputs into filtered, labelled detections.

        Each query contributes at most its single best class. Emitting every
        class above the threshold for a query would report the same object
        several times under different labels.
        """
        best_class = scores.argmax(axis=-1)
        best_score = scores.max(axis=-1)

        keep = np.where(best_score >= threshold)[0]
        if keep.size == 0:
            return []

        # Highest scoring first, then truncate: the caller's cap must keep the
        # most confident detections rather than an arbitrary 300-query slice.
        keep = keep[np.argsort(-best_score[keep])][:max_detections]

        absolute = boxes_to_absolute(boxes[keep], width, height)

        detections: list[Detection] = []
        for index, box in zip(keep, absolute, strict=True):
            class_id = int(best_class[index])
            detections.append(
                Detection(
                    label=(
                        model.class_names[class_id]
                        if class_id < len(model.class_names)
                        else str(class_id)
                    ),
                    class_id=class_id,
                    confidence=round(float(best_score[index]), 6),
                    box=BoundingBox(
                        x_min=round(float(box[0]), 2),
                        y_min=round(float(box[1]), 2),
                        x_max=round(float(box[2]), 2),
                        y_max=round(float(box[3]), 2),
                    ),
                )
            )

        return detections
