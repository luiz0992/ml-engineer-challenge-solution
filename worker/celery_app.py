"""Celery application for background batch inference.

Batch jobs run out-of-process because a large batch can take minutes, which is
far longer than an HTTP request should hold a connection open. The API
validates and enqueues; the worker does the work and records results.

Configuration notes, each addressing a specific failure mode:

* ``task_acks_late`` — a task is acknowledged after completion, not on receipt,
  so a worker killed mid-job returns the task to the queue instead of losing
  it silently.
* ``worker_prefetch_multiplier = 1`` — inference tasks are long and uneven.
  The default prefetch lets one worker reserve a queue of tasks it cannot start
  while another sits idle.
* ``task_time_limit`` — a hard ceiling, so a pathological job cannot occupy a
  worker indefinitely.
* ``result_expires`` — results are cleaned up rather than accumulating in Redis
  forever.
"""

from __future__ import annotations

from celery import Celery

from api.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ml_api",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=900,
    task_soft_time_limit=840,
    result_expires=86400,
    broker_connection_retry_on_startup=True,
)
