"""Celery tasks for batch inference.

Images are fetched by URL. Two properties matter:

* **One bad item does not fail the job.** Each item is fetched and processed
  independently and records either a result or an error, so a single dead URL
  in a thousand-item batch does not discard 999 successful predictions.
* **Fetching is bounded.** Every request has a timeout and a size cap, applied
  while streaming rather than after, so a hostile or misconfigured URL cannot
  hang a worker or exhaust its memory.

The model is loaded once per worker process and reused. Loading per task would
add hundreds of milliseconds to every item and defeat the purpose of batching.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import httpx
from celery import Task

from api.config import get_settings
from api.exceptions import APIError
from api.logging_config import configure_logging, get_logger
from api.services.cache_service import CacheService
from api.services.inference_service import InferenceService, summarise_predictions
from api.services.model_service import ModelService
from worker.celery_app import celery_app

logger = get_logger(__name__)

FETCH_TIMEOUT_SECONDS = 20.0
#: Chunk size for streamed downloads.
FETCH_CHUNK_BYTES = 64 * 1024


@lru_cache(maxsize=1)
def _get_services() -> tuple[InferenceService, ModelService]:
    """Build the inference stack once per worker process.

    Cached because Celery forks worker processes and each handles many tasks;
    rebuilding sessions per task would dominate runtime. The worker runs
    without a cache backend: batch items are typically unique, so a result
    cache would add Redis traffic for almost no hits.
    """
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_format == "json")

    model_service = ModelService(settings)
    model_service.load_classifier()

    cache = CacheService(None)
    return InferenceService(model_service, cache, settings), model_service


def _fetch_image(client: httpx.Client, url: str, max_bytes: int) -> bytes:
    """Download an image, enforcing the size cap during the transfer."""
    with client.stream("GET", url, timeout=FETCH_TIMEOUT_SECONDS) as response:
        response.raise_for_status()

        # Trust but verify: Content-Length is a hint, so the running total is
        # still checked as bytes arrive.
        declared = response.headers.get("content-length")
        if declared is not None and int(declared) > max_bytes:
            raise ValueError(
                f"Image is {int(declared)} bytes, exceeding the {max_bytes} byte limit"
            )

        buffer = bytearray()
        for chunk in response.iter_bytes(FETCH_CHUNK_BYTES):
            buffer.extend(chunk)
            if len(buffer) > max_bytes:
                raise ValueError(f"Image exceeds the {max_bytes} byte limit")

    return bytes(buffer)


@celery_app.task(bind=True, name="worker.tasks.process_batch")
def process_batch(
    self: Task,
    items: list[dict[str, Any]],
    *,
    task_type: str = "classification",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process a batch of images by URL.

    Returns a summary with a per-item result or error. Progress is published to
    the result backend as items complete, so a client polling the status
    endpoint sees movement rather than an opaque wait.
    """
    options = options or {}
    inference, _ = _get_services()
    settings = get_settings()

    started_at = datetime.now(UTC)
    results: list[dict[str, Any]] = []
    completed = failed = 0

    # follow_redirects is off: an allowed URL that redirects to an internal
    # address would otherwise bypass the scheme validation done at submission.
    with httpx.Client(follow_redirects=False) as client:
        for index, item in enumerate(items):
            url = item["image_url"]
            item_id = item.get("item_id")

            try:
                image_bytes = _fetch_image(client, url, settings.max_upload_bytes)
                predictions = asyncio.run(
                    inference.classify_many(
                        [image_bytes],
                        top_k_results=options.get("top_k", 5),
                        model_version=options.get("model_version"),
                    )
                )
                results.append(
                    {
                        "item_id": item_id,
                        "image_url": url,
                        "status": "succeeded",
                        "result": summarise_predictions(predictions[0]),
                        "error": None,
                    }
                )
                completed += 1

            except Exception as exc:
                failed += 1
                results.append(
                    {
                        "item_id": item_id,
                        "image_url": url,
                        "status": "failed",
                        "result": None,
                        "error": _describe_error(exc),
                    }
                )
                logger.warning(
                    "batch_item_failed",
                    job_id=self.request.id,
                    url=url,
                    error=str(exc),
                )

            self.update_state(
                state="PROGRESS",
                meta={
                    "total_items": len(items),
                    "completed_items": completed,
                    "failed_items": failed,
                    "current_index": index + 1,
                },
            )

    finished_at = datetime.now(UTC)
    logger.info(
        "batch_complete",
        job_id=self.request.id,
        total=len(items),
        completed=completed,
        failed=failed,
        duration_s=round((finished_at - started_at).total_seconds(), 2),
    )

    return {
        "task": task_type,
        "total_items": len(items),
        "completed_items": completed,
        "failed_items": failed,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "results": results,
    }


def _describe_error(exc: Exception) -> dict[str, Any]:
    """Render an exception as a caller-safe error object.

    Application errors carry a curated message; anything else is reported
    generically, because an arbitrary exception string can contain internal
    paths or configuration.
    """
    if isinstance(exc, APIError):
        return {"code": exc.code, "message": exc.message}
    if isinstance(exc, httpx.HTTPStatusError):
        return {
            "code": "fetch_failed",
            "message": f"The image URL returned HTTP {exc.response.status_code}.",
        }
    if isinstance(exc, httpx.RequestError):
        return {"code": "fetch_failed", "message": "The image URL could not be reached."}
    if isinstance(exc, ValueError):
        return {"code": "invalid_image", "message": str(exc)}

    logger.exception("batch_item_unexpected_error")
    return {"code": "internal_error", "message": "The item could not be processed."}
