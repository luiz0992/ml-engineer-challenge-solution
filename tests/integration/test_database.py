"""Database operations against a real Postgres.

Scope is deliberately narrow: only behaviour that a fake cannot demonstrate.
Anything provable with a stand-in belongs in `tests/unit/test_audit_service.py`,
which is faster and does not need Docker. See `conftest.py` in this directory
for why these three areas need the real thing.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from api.db.models import InferenceLog
from api.db.session import check_connection, session_scope
from api.services.audit_service import AuditService

pytestmark = pytest.mark.integration


def make_record(**overrides: Any) -> dict[str, Any]:
    """A valid audit record, with fields overridable per test."""
    record: dict[str, Any] = {
        "correlation_id": "abc123",
        "user_id": "user-1",
        "user_tier": "pro",
        "model_name": "classifier",
        "model_version": "v1",
        "backend": "onnx",
        "task": "classification",
        "status": "success",
        "latency_ms": 12.5,
        "cached": False,
        "batch_size": 1,
        "image_sha256": "a" * 64,
        "image_bytes": 2048,
        "image_width": 224,
        "image_height": 224,
        "image_format": "JPEG",
        "top_label": "tabby",
        "top_class_id": 7,
        "top_probability": 0.91,
    }
    record.update(overrides)
    return record


async def count_rows(factory: async_sessionmaker[AsyncSession]) -> int:
    async with factory() as session:
        result = await session.execute(select(func.count()).select_from(InferenceLog))
        return int(result.scalar_one())


class TestConnectivity:
    async def test_check_connection_reports_healthy(self, pg_engine: AsyncEngine) -> None:
        healthy, detail = await check_connection(pg_engine)

        assert healthy is True
        assert detail is None

    async def test_session_scope_commits(
        self, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with session_scope(pg_sessions) as session:
            session.add(InferenceLog(**make_record()))

        assert await count_rows(pg_sessions) == 1

    async def test_session_scope_rolls_back_on_error(
        self, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        with pytest.raises(RuntimeError):
            async with session_scope(pg_sessions) as session:
                session.add(InferenceLog(**make_record()))
                raise RuntimeError("boom")

        assert await count_rows(pg_sessions) == 0


class TestSchema:
    """The DDL the ORM emits, as Postgres actually applied it."""

    async def test_created_at_round_trips_timezone_aware(
        self, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        """A naive timestamp is ambiguous the moment the app and database differ.

        The column is `DateTime(timezone=True)`; whether the value comes back
        aware depends on the driver, so it is worth asserting against the real
        one.
        """
        async with session_scope(pg_sessions) as session:
            session.add(InferenceLog(**make_record()))

        async with pg_sessions() as session:
            row = (await session.execute(select(InferenceLog))).scalar_one()

        assert row.created_at.tzinfo is not None
        assert abs(datetime.now(UTC) - row.created_at) < timedelta(minutes=5)

    async def test_server_default_timestamps_rows_inserted_outside_the_orm(
        self, pg_engine: AsyncEngine, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        """`server_default=func.now()` exists so psql inserts are timestamped too.

        The Python-side `default` would cover the ORM path on its own, which is
        why this asserts through raw SQL instead.
        """
        async with pg_engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO inference_logs "
                    "(correlation_id, model_name, model_version, backend, task, "
                    " status, latency_ms, cached, batch_size) "
                    "VALUES ('raw', 'classifier', 'v1', 'onnx', 'classification', "
                    "        'success', 1.0, false, 1)"
                )
            )

        async with pg_sessions() as session:
            row = (await session.execute(select(InferenceLog))).scalar_one()

        assert row.created_at is not None
        assert row.created_at.tzinfo is not None

    @pytest.mark.parametrize(
        "index_name",
        [
            "ix_inference_logs_created_at",
            "ix_inference_logs_model_created",
            "ix_inference_logs_user_created",
            "ix_inference_logs_errors",
            "ix_inference_logs_image_sha256",
            "ix_inference_logs_variant",
        ],
    )
    async def test_index_exists(self, pg_engine: AsyncEngine, index_name: str) -> None:
        """Every declared index reached the database.

        Two of these are partial and two are descending. A typo in a
        `postgresql_where` clause produces a valid model and a silently missing
        index, and the first symptom is a slow query months later.
        """
        async with pg_engine.connect() as connection:
            result = await connection.execute(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"),
                {"name": index_name},
            )
            definition = result.scalar_one_or_none()

        assert definition is not None, f"{index_name} was never created"

    async def test_partial_indexes_carry_their_predicate(self, pg_engine: AsyncEngine) -> None:
        """The partial indexes are partial.

        Without the WHERE clause they still work, but they index every row and
        become the largest indexes on an append-heavy table -- the exact cost
        the model comments say they avoid.
        """
        async with pg_engine.connect() as connection:
            result = await connection.execute(
                text(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE indexname IN "
                    "('ix_inference_logs_errors', 'ix_inference_logs_variant')"
                )
            )
            definitions: dict[str, str] = dict(result.all())  # type: ignore[arg-type]

        assert "WHERE" in definitions["ix_inference_logs_errors"]
        assert "WHERE" in definitions["ix_inference_logs_variant"]


class TestAuditServiceAgainstPostgres:
    async def test_records_are_written_and_readable(
        self, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        service = AuditService(pg_sessions)
        await service.start()
        try:
            service.record(**make_record(correlation_id="written-1"))
            await service.flush()
        finally:
            await service.stop()

        async with pg_sessions() as session:
            row = (await session.execute(select(InferenceLog))).scalar_one()

        assert row.correlation_id == "written-1"
        assert row.model_name == "classifier"
        assert row.top_probability == pytest.approx(0.91)
        assert service.stats.written == 1
        assert service.stats.failed == 0

    async def test_batches_larger_than_one_insert_are_all_persisted(
        self, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        service = AuditService(pg_sessions, batch_size=10)
        await service.start()
        try:
            for i in range(25):
                service.record(**make_record(correlation_id=f"batch-{i}"))
            await service.flush()
        finally:
            await service.stop()

        assert await count_rows(pg_sessions) == 25

    async def test_stop_drains_records_still_queued(
        self, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        """The property that keeps audit gaps from clustering around deploys.

        A long flush interval means nothing has been written when `stop` is
        called, so this fails if the drain is dropped.
        """
        service = AuditService(pg_sessions, flush_interval=3600.0)
        await service.start()
        for i in range(5):
            service.record(**make_record(correlation_id=f"drain-{i}"))

        assert await count_rows(pg_sessions) == 0

        await service.stop()

        assert await count_rows(pg_sessions) == 5

    async def test_a_rejected_write_is_counted_rather_than_raised(
        self, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        """`_write` swallows exceptions by contract; prove it still notices.

        `correlation_id` is `String(64)` and NOT NULL. A null violates the
        constraint in the database rather than in Python, so this exercises the
        real failure path -- and distinguishes "failed and was counted" from
        "silently did nothing", which are indistinguishable against a fake.
        """
        service = AuditService(pg_sessions)
        await service.start()
        try:
            service.record(**make_record(correlation_id=None))
            await service.flush()
        finally:
            await service.stop()

        assert service.stats.failed == 1
        assert service.stats.written == 0
        assert await count_rows(pg_sessions) == 0

    async def test_one_bad_batch_does_not_stop_later_writes(
        self, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        """A poisoned batch must not wedge the writer for the rest of the process."""
        service = AuditService(pg_sessions)
        await service.start()
        try:
            service.record(**make_record(correlation_id=None))
            await service.flush()

            service.record(**make_record(correlation_id="after-failure"))
            await service.flush()
        finally:
            await service.stop()

        async with pg_sessions() as session:
            row = (await session.execute(select(InferenceLog))).scalar_one()

        assert row.correlation_id == "after-failure"
        assert service.stats.failed == 1
        assert service.stats.written == 1

    async def test_concurrent_writers_do_not_lose_records(
        self, pg_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        """Several requests audit at once; the pool must serve them all.

        `pool_size=10` with `pool_timeout=5` means an exhausted pool raises
        rather than queueing, which would surface here as missing rows.
        """
        service = AuditService(pg_sessions, batch_size=5)
        await service.start()
        try:

            async def submit(worker: int) -> None:
                for i in range(10):
                    service.record(**make_record(correlation_id=f"w{worker}-{i}"))
                    await asyncio.sleep(0)

            await asyncio.gather(*(submit(w) for w in range(8)))
            await service.flush()
        finally:
            await service.stop()

        assert await count_rows(pg_sessions) == 80
