"""Tests for inference audit logging.

The behaviours that matter are what happens when the database misbehaves. An
audit trail is valuable, but never more valuable than serving the request, so
every degradation path is asserted explicitly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from api.services.audit_service import AuditService

pytestmark = pytest.mark.unit


def _record(**overrides: Any) -> dict[str, Any]:
    """A minimal valid audit record."""
    base = {
        "correlation_id": "abc123",
        "user_id": "alice",
        "user_tier": "pro",
        "model_name": "classifier",
        "model_version": "v1",
        "backend": "onnx",
        "task": "classification",
        "status": "success",
        "latency_ms": 12.5,
        "cached": False,
        "batch_size": 1,
    }
    return {**base, **overrides}


class _CapturingSessionFactory:
    """A session factory that records what would have been written."""

    def __init__(self, *, fail: bool = False) -> None:
        self.written: list[Any] = []
        self.fail = fail
        self.commits = 0

    def __call__(self) -> Any:
        return self

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def add_all(self, objects: list[Any]) -> None:
        self.written.extend(objects)

    async def commit(self) -> None:
        if self.fail:
            raise RuntimeError("database is unreachable")
        self.commits += 1


class TestDisabledService:
    """Without a database the service must be completely inert."""

    async def test_record_is_a_no_op(self) -> None:
        service = AuditService(None)
        service.record(**_record())

        assert not service.enabled
        assert service.stats.queued == 0

    async def test_start_and_stop_are_safe(self) -> None:
        """Lifecycle calls must be no-ops, not errors, with no database.

        The API calls start() and stop() unconditionally, so a disabled audit
        service raising here would fail startup for a component that is
        explicitly optional.
        """
        service = AuditService(None)

        await service.start()
        await service.stop()

        # No background task was created, and no work was recorded.
        assert service._task is None
        assert service.stats.written == 0


class TestBuffering:
    async def test_records_are_written_in_batches(self) -> None:
        factory = _CapturingSessionFactory()
        service = AuditService(factory, batch_size=5, flush_interval=0.05)
        await service.start()

        for index in range(5):
            service.record(**_record(correlation_id=f"req-{index}"))

        await asyncio.sleep(0.2)
        await service.stop()

        assert len(factory.written) == 5
        # One commit for five records: batching is what keeps the audit trail
        # from costing a round trip per inference.
        assert factory.commits == 1
        assert service.stats.written == 5

    async def test_partial_batch_is_flushed_on_a_timer(self) -> None:
        """A quiet service must still persist its records promptly."""
        factory = _CapturingSessionFactory()
        service = AuditService(factory, batch_size=100, flush_interval=0.05)
        await service.start()

        service.record(**_record())
        await asyncio.sleep(0.2)

        assert len(factory.written) == 1
        await service.stop()

    async def test_shutdown_drains_outstanding_records(self) -> None:
        """Records buffered at shutdown must not be lost.

        Without draining, every record held at the moment of a rolling deploy
        is discarded, so audit gaps would cluster exactly around deploys --
        which is when the trail is most needed.
        """
        factory = _CapturingSessionFactory()
        service = AuditService(factory, batch_size=1000, flush_interval=60.0)
        await service.start()

        for index in range(10):
            service.record(**_record(correlation_id=f"req-{index}"))

        # Stop before the flush interval could possibly have elapsed.
        await service.stop()

        assert len(factory.written) == 10


class TestFailsOpen:
    async def test_record_never_raises_when_the_database_fails(self) -> None:
        factory = _CapturingSessionFactory(fail=True)
        service = AuditService(factory, batch_size=2, flush_interval=0.05)
        await service.start()

        for index in range(4):
            service.record(**_record(correlation_id=f"req-{index}"))

        await asyncio.sleep(0.2)
        await service.stop()

        # The failure is counted rather than raised, so the request path is
        # unaffected.
        assert service.stats.failed > 0
        assert service.stats.written == 0

    async def test_record_is_non_blocking(self) -> None:
        """Recording must not be awaited, so it cannot add request latency."""
        service = AuditService(_CapturingSessionFactory(), batch_size=10)

        # A synchronous call: if record() were a coroutine this would return an
        # un-awaited object rather than queueing anything, and the request path
        # would have to await the audit trail.
        service.record(**_record())

        assert service.stats.queued == 1
        assert not asyncio.iscoroutinefunction(service.record)


class TestBoundedQueue:
    async def test_queue_cannot_grow_without_bound(self) -> None:
        """A database outage must not become an OOM kill.

        The writer is never started, so nothing drains the queue -- simulating
        a database that is unreachable for a sustained period.
        """
        service = AuditService(_CapturingSessionFactory(), queue_size=10)

        for index in range(100):
            service.record(**_record(correlation_id=f"req-{index}"))

        assert service._queue.qsize() <= 10
        assert service.stats.dropped > 0

    async def test_oldest_records_are_dropped_first(self) -> None:
        """The newest records survive, since they matter most in an incident."""
        service = AuditService(_CapturingSessionFactory(), queue_size=5)

        for index in range(20):
            service.record(**_record(correlation_id=f"req-{index}"))

        remaining = []
        while not service._queue.empty():
            remaining.append(service._queue.get_nowait()["correlation_id"])

        assert "req-19" in remaining
        assert "req-0" not in remaining


class TestStats:
    async def test_counters_reflect_activity(self) -> None:
        factory = _CapturingSessionFactory()
        service = AuditService(factory, batch_size=2, flush_interval=0.05)
        await service.start()

        for index in range(4):
            service.record(**_record(correlation_id=f"req-{index}"))

        await asyncio.sleep(0.2)
        await service.stop()

        stats = service.stats.as_dict()
        assert stats["queued"] == 4
        assert stats["written"] == 4
        assert stats["dropped"] == 0
        assert stats["failed"] == 0


class TestIntegrationWithInference:
    async def test_successful_inference_is_recorded(
        self, model_service: Any, cache_service: Any, settings: Any
    ) -> None:
        from api.services.inference_service import InferenceService
        from tests.conftest import make_image_bytes

        factory = _CapturingSessionFactory()
        audit = AuditService(factory, batch_size=1, flush_interval=0.05)
        service = InferenceService(model_service, cache_service, settings, audit)

        await service.classify(
            make_image_bytes(),
            correlation_id="trace-1",
            user_id="alice",
            user_tier="pro",
            use_cache=False,
        )
        await audit.flush()

        assert len(factory.written) == 1
        row = factory.written[0]
        assert row.correlation_id == "trace-1"
        assert row.user_id == "alice"
        assert row.status == "success"
        assert row.top_label is not None
        # A digest of the upload itself, not of its metadata: a fingerprint
        # derived from dimensions would collide for every same-sized image.
        assert len(row.image_sha256) == 64

    async def test_cache_hits_are_recorded_and_marked(
        self, model_service: Any, cache_service: Any, settings: Any
    ) -> None:
        """A cached response is still a served prediction and must be audited."""
        from api.services.inference_service import InferenceService
        from tests.conftest import make_image_bytes

        factory = _CapturingSessionFactory()
        audit = AuditService(factory, batch_size=1, flush_interval=0.05)
        service = InferenceService(model_service, cache_service, settings, audit)
        image = make_image_bytes(seed=5)

        await service.classify(image, correlation_id="first")
        await service.classify(image, correlation_id="second")
        await audit.flush()

        assert len(factory.written) == 2
        assert factory.written[0].cached is False
        assert factory.written[1].cached is True

    async def test_inference_succeeds_without_an_audit_service(
        self, model_service: Any, cache_service: Any, settings: Any
    ) -> None:
        """Auditing is optional; the worker runs without it."""
        from api.services.inference_service import InferenceService
        from tests.conftest import make_image_bytes

        service = InferenceService(model_service, cache_service, settings)
        response = await service.classify(make_image_bytes(), correlation_id="c")

        assert response.predictions
