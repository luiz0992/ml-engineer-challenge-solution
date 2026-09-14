"""Shared FastAPI dependencies.

Services are constructed once during application startup and stored on
``app.state``. These dependencies read them back, so request handlers declare
what they need without knowing how it was built — which is also what makes them
trivial to override in tests.
"""

from __future__ import annotations

from fastapi import Depends, Request

from api.config import Settings, get_settings
from api.exceptions import ModelUnavailableError, ServiceUnavailableError
from api.middleware.rate_limit import RateLimiter
from api.services.cache_service import CacheService
from api.services.detection_service import DetectionService
from api.services.inference_service import InferenceService
from api.services.model_service import ModelService
from api.services.similarity_service import SimilarityService


def get_model_service(request: Request) -> ModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise ServiceUnavailableError("The model service is not initialised.")
    return service


def get_cache_service(request: Request) -> CacheService:
    service = getattr(request.app.state, "cache_service", None)
    if service is None:
        # The cache is optional by design; an inert instance keeps call sites
        # free of None checks and preserves fail-open behaviour.
        return CacheService(None)
    return service


def get_inference_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise ServiceUnavailableError("The inference service is not initialised.")
    return service


def get_detection_service(request: Request) -> DetectionService:
    """Return the detection service, or 503 when no detector is deployed.

    Detection is optional: the service is useful without it, so a missing
    detector degrades one endpoint rather than failing startup.
    """
    service = getattr(request.app.state, "detection_service", None)
    if service is None:
        raise ModelUnavailableError(
            "No detection model is deployed in this environment. Classification "
            "is available at POST /api/v1/classify."
        )
    return service


def get_similarity_service(request: Request) -> SimilarityService:
    """Return the similarity service, or 503 when no index is built."""
    service = getattr(request.app.state, "similarity_service", None)
    if service is None:
        raise ModelUnavailableError(
            "Similarity search is not available in this environment. Build the "
            "index with `python scripts/build_similarity_index.py`."
        )
    return service


def get_rate_limiter(request: Request) -> RateLimiter:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        raise ServiceUnavailableError("The rate limiter is not initialised.")
    return limiter


def get_correlation_id(request: Request) -> str:
    return getattr(request.state, "correlation_id", "unknown")


def get_app_settings(settings: Settings = Depends(get_settings)) -> Settings:
    return settings
