"""Image classification endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile

from api.config import InferenceBackend
from api.dependencies import (
    get_correlation_id,
    get_experiment_registry,
    get_inference_service,
    get_rate_limiter,
)
from api.exceptions import PayloadTooLargeError
from api.middleware.auth import Principal, authenticate
from api.middleware.monitoring import record_inference
from api.middleware.rate_limit import RateLimiter
from api.models.responses import ClassificationResponse, ErrorResponse
from api.services.experiment_service import ExperimentRegistry
from api.services.inference_service import InferenceService

router = APIRouter(prefix="/classify", tags=["classification"])


@router.post(
    "",
    response_model=ClassificationResponse,
    summary="Classify an image",
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid token"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
        422: {"model": ErrorResponse, "description": "Invalid or unsupported image"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "No model backend available"},
    },
)
async def classify_image(
    # Named to avoid shadowing the ClassificationResponse built below.
    http_response: Response,
    file: UploadFile = File(description="Image to classify (JPEG, PNG, WebP, or BMP)."),
    top_k: int = Query(default=5, ge=1, le=100, description="Predictions to return."),
    model_version: str | None = Query(
        default=None, description="Pin a model version. Defaults to the active one."
    ),
    backend: InferenceBackend | None = Query(
        default=None,
        description=(
            "Pin an inference runtime (onnx, onnx-int8, tensorrt). Ignored unless "
            "ALLOW_BACKEND_OVERRIDE is enabled. INT8 is measured and available; "
            "it is not the default because it costs 5.4pp of top-1."
        ),
    ),
    include_probabilities: bool = Query(default=True),
    use_cache: bool = Query(default=True, description="Set false to bypass the result cache."),
    principal: Principal = Depends(authenticate),
    inference: InferenceService = Depends(get_inference_service),
    limiter: RateLimiter = Depends(get_rate_limiter),
    experiments: ExperimentRegistry = Depends(get_experiment_registry),
    correlation_id: str = Depends(get_correlation_id),
) -> ClassificationResponse:
    """Classify a single image and return ranked predictions.

    Requires a bearer token. Rate limits apply per user tier and are reported
    in the ``X-RateLimit-*`` response headers.
    """
    decision = await limiter.enforce(principal.user_id, principal.tier)
    # Returned on success as well as rejection, so clients can self-throttle
    # before they are rejected rather than discovering the limit by hitting it.
    http_response.headers.update(decision.headers)

    # An explicitly pinned version always wins over an experiment: a caller
    # asking for a specific version must get it, or version pinning is a lie.
    variant_name: str | None = None
    if model_version is None:
        model_version, variant_name = experiments.resolve_version(
            "tiny-imagenet-classifier", principal.user_id
        )

    image_bytes = await _read_upload(file, inference.settings.max_upload_bytes)

    started = time.perf_counter()
    try:
        response = await inference.classify(
            image_bytes,
            correlation_id=correlation_id,
            top_k_results=top_k,
            model_version=model_version,
            backend=backend if inference.settings.allow_backend_override else None,
            include_probabilities=include_probabilities,
            use_cache=use_cache,
            user_id=principal.user_id,
            user_tier=principal.tier.value,
            variant=variant_name,
        )
    except Exception:
        model = inference.models.get_classifier(model_version)
        record_inference(
            model=model.name,
            version=model.version,
            backend=model.backend.value,
            duration_seconds=time.perf_counter() - started,
            success=False,
        )
        raise

    record_inference(
        model=response.provenance.model_name,
        version=response.provenance.model_version,
        backend=response.provenance.backend,
        duration_seconds=time.perf_counter() - started,
        success=True,
    )
    return response


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, aborting once it exceeds the limit.

    Read in chunks and stop at the threshold rather than calling ``read()``.
    An unbounded read buffers the entire body first, so a multi-gigabyte upload
    would exhaust memory before any size check could reject it — the check has
    to happen *during* the read, not after.
    """
    chunk_size = 1024 * 1024
    buffer = bytearray()

    while chunk := await file.read(chunk_size):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise PayloadTooLargeError(
                f"Upload exceeds the {max_bytes / 1024 / 1024:.1f} MiB limit.",
                details={"max_bytes": max_bytes},
            )

    return bytes(buffer)
