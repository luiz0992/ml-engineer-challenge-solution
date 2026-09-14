"""Inference audit logging.

Records every inference to Postgres for provenance, drift analysis, and
investigation.

Two properties govern the design:

**A database problem must never fail a request.** The inference succeeded; the
user is entitled to their answer whether or not we managed to record it. Every
write is wrapped, and failures are logged and counted rather than raised.

**Recording must not add latency to the response.** Writes are queued and
flushed by a background task rather than awaited inline. A synchronous insert
would add a network round trip to a request whose entire budget is a few
milliseconds — the audit trail would cost more than the inference.

The queue is bounded. An unbounded queue turns a database outage into
unbounded memory growth and eventually an OOM kill, converting a degraded
service into a dead one. When full, the oldest entries are dropped and the loss
is counted, because recent records are more useful than old ones during an
incident.

**Scope: inference attempts, not all requests.** A request rejected during
input validation produces no row, because validation runs before a model is
resolved and there is no provenance to attribute the record to. Those are
client errors and are visible in the HTTP metrics (`http_requests_total` by
status code); this table answers "what did which model predict", which a
request that never reached a model cannot contribute to. Failures that occur
*during* inference are recorded, since an error rate computed only from
successful rows would be meaningless.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

from api.db.models import InferenceLog
from api.logging_config import get_logger

logger = get_logger(__name__)

#: Bounded so a database outage cannot exhaust memory.
DEFAULT_QUEUE_SIZE = 10_000

#: Rows per INSERT. Batching amortises the round trip; too large a batch holds
#: rows in memory longer and lengthens the window in which a crash loses them.
DEFAULT_BATCH_SIZE = 50

#: Maximum time a record waits before being written, so a quiet service still
#: persists its audit trail promptly.
DEFAULT_FLUSH_INTERVAL_SECONDS = 2.0


@dataclass(slots=True)
class AuditStats:
    """Counters exposed through the health endpoint."""

    queued: int = 0
    written: int = 0
    dropped: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "queued": self.queued,
            "written": self.written,
            "dropped": self.dropped,
            "failed": self.failed,
        }


class AuditService:
    """Buffers inference records and writes them in batches."""

    def __init__(
        self,
        # Typed as a callable rather than async_sessionmaker specifically: the
        # service only ever calls it and uses the result as an async context
        # manager, so requiring the concrete SQLAlchemy type would exclude any
        # equivalent -- including the capturing double the tests use, which is
        # precisely the substitutability the narrow type was preventing.
        session_factory: Callable[[], AbstractAsyncContextManager[Any]] | None,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self._session_factory = session_factory
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._task: asyncio.Task[None] | None = None
        self.stats = AuditStats()

    @property
    def enabled(self) -> bool:
        return self._session_factory is not None

    async def start(self) -> None:
        """Start the background writer."""
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="audit-writer")
        logger.info("audit_writer_started", batch_size=self._batch_size)

    async def stop(self) -> None:
        """Flush outstanding records and stop the writer.

        Draining on shutdown matters: without it, every record buffered at the
        moment of a rolling deploy is lost, and audit gaps cluster precisely
        around deploys, which is when they are most needed.
        """
        if self._task is None:
            return

        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

        await self._drain()
        logger.info("audit_writer_stopped", **self.stats.as_dict())

    def record(self, **fields: Any) -> None:
        """Queue one inference record.

        Synchronous and non-blocking by design, so a caller never awaits the
        audit trail. A full queue drops the oldest record rather than blocking
        the request or growing without bound.
        """
        if not self.enabled:
            return

        try:
            self._queue.put_nowait(fields)
            self.stats.queued += 1
        except asyncio.QueueFull:
            # Make room by discarding the oldest, then retry once. During an
            # outage this keeps the most recent records, which are the ones an
            # investigation wants.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self.stats.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(fields)
                self.stats.queued += 1

    async def _run(self) -> None:
        """Collect records and flush them in batches."""
        batch: list[dict[str, Any]] = []

        while True:
            try:
                # Wait for the first record, then drain whatever else is ready.
                # This gives low latency when busy and no busy-wait when idle.
                first = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
                batch.append(first)

                while len(batch) < self._batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                raise

            if batch:
                await self._write(batch)
                batch = []

    async def _drain(self) -> None:
        """Write everything still queued."""
        batch: list[dict[str, Any]] = []
        while not self._queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                batch.append(self._queue.get_nowait())
            if len(batch) >= self._batch_size:
                await self._write(batch)
                batch = []
        if batch:
            await self._write(batch)

    async def _write(self, batch: list[dict[str, Any]]) -> None:
        """Insert a batch, never raising.

        A failure here is logged and counted. Retrying would risk an unbounded
        loop against a database that is down, and the records are not valuable
        enough to justify blocking the writer.
        """
        if self._session_factory is None:
            return

        try:
            async with self._session_factory() as session:
                session.add_all([InferenceLog(**fields) for fields in batch])
                await session.commit()
            self.stats.written += len(batch)
        except Exception as exc:
            self.stats.failed += len(batch)
            logger.warning(
                "audit_write_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                records=len(batch),
            )

    async def flush(self) -> None:
        """Write everything queued. Used by tests and shutdown."""
        await self._drain()
