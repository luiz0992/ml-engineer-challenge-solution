"""Object detection endpoints.

The detection model is registered but not yet wired to a loaded backend. The
endpoint is defined so the API contract is complete and documented, and returns
503 with an explicit explanation rather than 404 or a silent stub — a caller
must be able to tell "not deployed here" apart from "you called the wrong URL".
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile

from api.dependencies import get_correlation_id, get_model_service, get_rate_limiter
from api.exceptions import ModelUnavailableError
from api.middleware.auth import Principal, authenticate
from api.middleware.rate_limit import RateLimiter
from api.models.responses import DetectionResponse, ErrorResponse
from api.models.schemas import TaskType
from api.services.model_service import ModelService

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post(
    "",
    response_model=DetectionResponse,
    summary="Detect objects in an image",
    responses={
        401: {"model": ErrorResponse},
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
    confidence_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    max_detections: int = Query(default=100, ge=1, le=1000),
    model_version: str | None = Query(default=None),
    principal: Principal = Depends(authenticate),
    models: ModelService = Depends(get_model_service),
    limiter: RateLimiter = Depends(get_rate_limiter),
    correlation_id: str = Depends(get_correlation_id),
) -> DetectionResponse:
    """Detect objects and return labelled bounding boxes.

    Boxes are in absolute pixel coordinates of the uploaded image.
    """
    decision = await limiter.enforce(principal.user_id, principal.tier)
    response.headers.update(decision.headers)

    detection_models = [m for m in models.list_models() if m.task is TaskType.DETECTION]
    if not detection_models:
        raise ModelUnavailableError(
            "No detection model is deployed in this environment. Classification "
            "is available at POST /api/v1/classify.",
            details={"available_tasks": sorted({m.task.value for m in models.list_models()})},
        )

    # Unreachable until a detection model is registered; the loader raises above.
    raise ModelUnavailableError("The detection model is registered but not loadable.")
