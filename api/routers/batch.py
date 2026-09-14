"""Asynchronous batch processing endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, Request, Response, status

from api.config import Settings, get_settings
from api.dependencies import get_rate_limiter
from api.exceptions import BatchTooLargeError, JobNotFoundError
from api.logging_config import get_logger
from api.middleware.auth import Principal, authenticate
from api.middleware.rate_limit import RateLimiter
from api.models.responses import (
    BatchItemResult,
    BatchJobResponse,
    BatchJobStatusResponse,
    ErrorResponse,
)
from api.models.schemas import BatchRequest, JobStatus, TaskType

logger = get_logger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])

#: Celery state names mapped to the API's job lifecycle.
_STATE_MAP = {
    "PENDING": JobStatus.PENDING,
    "RECEIVED": JobStatus.PENDING,
    "STARTED": JobStatus.RUNNING,
    "PROGRESS": JobStatus.RUNNING,
    "RETRY": JobStatus.RUNNING,
    "SUCCESS": JobStatus.COMPLETED,
    "FAILURE": JobStatus.FAILED,
    "REVOKED": JobStatus.CANCELLED,
}


@router.post(
    "",
    response_model=BatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a batch job",
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse, "description": "Invalid request or batch too large"},
        429: {"model": ErrorResponse},
    },
)
async def submit_batch(
    payload: BatchRequest,
    request: Request,
    response: Response,
    principal: Principal = Depends(authenticate),
    limiter: RateLimiter = Depends(get_rate_limiter),
    settings: Settings = Depends(get_settings),
) -> BatchJobResponse:
    """Queue images for background processing.

    Returns 202 with a job ID immediately; the work happens in a Celery worker.
    Poll ``status_url`` for progress and results.
    """
    decision = await limiter.enforce(principal.user_id, principal.tier)
    response.headers.update(decision.headers)

    # Enforced here as well as in the service layer: rejecting an oversized
    # batch before it is enqueued avoids occupying a worker with a job that
    # would fail on its first item.
    max_items = settings.max_batch_size * 10
    if len(payload.items) > max_items:
        raise BatchTooLargeError(
            f"Batch contains {len(payload.items)} items; the maximum is {max_items}.",
            details={"submitted": len(payload.items), "max_items": max_items},
        )

    options: dict[str, object] = {}
    if payload.classification_options is not None:
        options.update(payload.classification_options.model_dump(exclude_none=True))

    from worker.tasks import process_batch

    async_result = process_batch.delay(
        [item.model_dump() for item in payload.items],
        task_type=payload.task.value,
        options=options,
    )

    logger.info(
        "batch_submitted",
        job_id=async_result.id,
        user_id=principal.user_id,
        items=len(payload.items),
        task=payload.task.value,
    )

    return BatchJobResponse(
        job_id=async_result.id,
        status=JobStatus.PENDING,
        task=payload.task,
        total_items=len(payload.items),
        submitted_at=datetime.now(UTC),
        status_url=str(request.url_for("get_batch_status", job_id=async_result.id)),
    )


@router.get(
    "/{job_id}",
    response_model=BatchJobStatusResponse,
    name="get_batch_status",
    summary="Check batch job status",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_batch_status(
    job_id: str,
    principal: Principal = Depends(authenticate),
) -> BatchJobStatusResponse:
    """Return progress and, once finished, per-item results."""
    result = AsyncResult(job_id)

    # Celery reports an unknown ID as PENDING, since it cannot distinguish a
    # job that has not started from one that never existed. Treating an unknown
    # ID as pending would leave a client polling forever for a typo, so a
    # PENDING job with no backend record is reported as not found.
    if result.state == "PENDING" and not result.backend.get_task_meta(job_id).get("status"):
        raise JobNotFoundError(
            f"No job found with ID {job_id!r}. It may have expired.",
            details={"job_id": job_id},
        )

    job_status = _STATE_MAP.get(result.state, JobStatus.PENDING)
    info = result.info if isinstance(result.info, dict) else {}

    if job_status is JobStatus.COMPLETED:
        payload = result.result or {}
        return BatchJobStatusResponse(
            job_id=job_id,
            status=job_status,
            task=TaskType(payload.get("task", TaskType.CLASSIFICATION.value)),
            total_items=payload.get("total_items", 0),
            completed_items=payload.get("completed_items", 0),
            failed_items=payload.get("failed_items", 0),
            submitted_at=_parse_time(payload.get("started_at")),
            started_at=_parse_time(payload.get("started_at")),
            finished_at=_parse_time(payload.get("finished_at")),
            results=[BatchItemResult(**item) for item in payload.get("results", [])],
        )

    if job_status is JobStatus.FAILED:
        return BatchJobStatusResponse(
            job_id=job_id,
            status=job_status,
            task=TaskType.CLASSIFICATION,
            total_items=info.get("total_items", 0),
            completed_items=info.get("completed_items", 0),
            failed_items=info.get("failed_items", 0),
            submitted_at=datetime.now(UTC),
            # The exception text is not returned: it can contain internal
            # detail. The full traceback is in the worker logs under this ID.
            error="The job failed. Contact support with this job ID.",
        )

    return BatchJobStatusResponse(
        job_id=job_id,
        status=job_status,
        task=TaskType.CLASSIFICATION,
        total_items=info.get("total_items", 0),
        completed_items=info.get("completed_items", 0),
        failed_items=info.get("failed_items", 0),
        submitted_at=datetime.now(UTC),
    )


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(UTC)
