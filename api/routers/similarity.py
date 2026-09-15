"""Image similarity search endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile

from api.dependencies import (
    get_correlation_id,
    get_rate_limiter,
    get_similarity_service,
)
from api.middleware.auth import Principal, authenticate
from api.middleware.monitoring import record_inference
from api.middleware.rate_limit import RateLimiter
from api.models.responses import ErrorResponse, SimilarityResponse
from api.routers.classification import _read_upload
from api.services.similarity_service import SimilarityService

router = APIRouter(prefix="/similar", tags=["similarity"])


@router.post(
    "",
    response_model=SimilarityResponse,
    summary="Find visually similar images",
    responses={
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {
            "model": ErrorResponse,
            "description": "Similarity index not built in this environment",
        },
    },
)
async def find_similar(
    response: Response,
    file: UploadFile = File(description="Query image."),
    top_k: int = Query(default=10, ge=1, le=100, description="Neighbours to return."),
    min_similarity: float = Query(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Discard results below this cosine similarity.",
    ),
    principal: Principal = Depends(authenticate),
    similarity: SimilarityService = Depends(get_similarity_service),
    limiter: RateLimiter = Depends(get_rate_limiter),
    correlation_id: str = Depends(get_correlation_id),
) -> SimilarityResponse:
    """Retrieve the most similar indexed images, ranked by cosine similarity.

    Results are nearest neighbours in the contrastive embedding space, ranked
    by cosine similarity.
    """
    decision = await limiter.enforce(principal.user_id, principal.tier)
    response.headers.update(decision.headers)

    image_bytes = await _read_upload(file, similarity.settings.max_upload_bytes)

    started = time.perf_counter()
    result = await similarity.find_similar(
        image_bytes,
        correlation_id=correlation_id,
        top_k=top_k,
        min_similarity=min_similarity,
    )

    record_inference(
        model=result.provenance.model_name,
        version=result.provenance.model_version,
        backend=result.provenance.backend,
        duration_seconds=time.perf_counter() - started,
        success=True,
    )
    return result
