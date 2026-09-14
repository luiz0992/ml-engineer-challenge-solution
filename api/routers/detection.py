"""Object detection endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile

from api.dependencies import get_correlation_id, get_detection_service, get_rate_limiter
from api.exceptions import ModelUnavailableError
from api.middleware.auth import Principal, authenticate
from api.middleware.monitoring import record_inference
from api.middleware.rate_limit import RateLimiter
from api.models.responses import DetectionResponse, ErrorResponse
from api.routers.classification import _read_upload
from api.services.detection_service import DetectionService

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post(
    "",
    response_model=DetectionResponse,
    summary="Detect objects in an image",
    responses={
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {
            "model": ErrorResponse,
            "description": "Detection model not deployed in this environment",
        },
    },
)
async def detect_objects(
    response: Response,
    file: UploadFile = File(description="Image to run detection on."),
    confidence_threshold: float = Query(
        default=0.5, ge=0.0, le=1.0, description="Discard detections below this score."
    ),
    max_detections: int = Query(
        default=100, ge=1, le=300, description="Cap on returned detections."
    ),
    model_version: str | None = Query(default=None),
    principal: Principal = Depends(authenticate),
    detection: DetectionService = Depends(get_detection_service),
    limiter: RateLimiter = Depends(get_rate_limiter),
    correlation_id: str = Depends(get_correlation_id),
) -> DetectionResponse:
    """Detect objects and return labelled bounding boxes.

    Boxes are in absolute pixel coordinates of the uploaded image, so a client
    can overlay them without knowing how the image was resized internally.

    Returns 503 rather than 404 when no detector is deployed: a caller must be
    able to distinguish "not available here" from "wrong URL".
    """
    decision = await limiter.enforce(principal.user_id, principal.tier)
    response.headers.update(decision.headers)

    image_bytes = await _read_upload(file, detection.settings.max_upload_bytes)

    started = time.perf_counter()
    try:
        result = await detection.detect(
            image_bytes,
            correlation_id=correlation_id,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
            model_version=model_version,
        )
    except ModelUnavailableError:
        raise
    except Exception:
        record_inference(
            model="rtdetr-coco-detector",
            version=model_version or "unknown",
            backend="unknown",
            duration_seconds=time.perf_counter() - started,
            success=False,
        )
        raise

    record_inference(
        model=result.provenance.model_name,
        version=result.provenance.model_version,
        backend=result.provenance.backend,
        duration_seconds=time.perf_counter() - started,
        success=True,
    )
    return result
