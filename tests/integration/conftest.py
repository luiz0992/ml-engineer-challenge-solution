"""Fixtures for tests that need a real Postgres.

Everything else in the suite runs against fakes — `fakeredis` for the cache, a
synthetic ONNX graph for the model — because they are faithful enough and
enormously faster. The database is the exception.

Three properties of the audit trail cannot be observed against a stand-in:

* `created_at` is `DateTime(timezone=True)` with `server_default=func.now()`.
  Whether a value comes back timezone-aware depends on the driver and the
  column type, not on our code, so a fake that returns whatever it was handed
  proves nothing.
* Four of the six indexes on `inference_logs` are Postgres-specific — two are
  partial (`postgresql_where`) and two are descending. A fake has no query
  planner and no DDL to get wrong.
* `AuditService._write` swallows every exception by contract. Against a fake,
  "the write failed and was counted" and "the write silently did nothing" look
  identical. Only a real constraint violation distinguishes them.

The container is session-scoped because starting Postgres costs a few seconds
and the tests are read-mostly; isolation comes from truncating between tests
rather than from a fresh database each time.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from api.config import AppEnv, Settings
from api.db.session import create_all, create_engine, create_session_factory

POSTGRES_IMAGE = "postgres:16-alpine"


@dataclass(frozen=True, slots=True)
class PostgresParams:
    """Connection parameters for the database under test."""

    host: str
    port: int
    user: str
    password: str
    database: str


@pytest.fixture(scope="session")
def postgres_params() -> Iterator[PostgresParams]:
    """A real Postgres to test against.

    Prefers a database supplied by the environment. CI already runs one as a
    service container, and starting a second inside the runner would be slower
    and would need a Docker socket we should not assume is there.

    Falls back to testcontainers for local runs, and skips -- rather than
    failing -- when neither is available, so `pytest` still works on a machine
    without Docker.
    """
    if host := os.getenv("POSTGRES_HOST"):
        yield PostgresParams(
            host=host,
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "mlapi"),
            password=os.getenv("POSTGRES_PASSWORD", "mlapi"),
            database=os.getenv("POSTGRES_DB", "mlapi"),
        )
        return

    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover - depends on the install extras
        pytest.skip("Postgres not available: testcontainers is not installed")

    try:
        with PostgresContainer(POSTGRES_IMAGE) as container:
            yield PostgresParams(
                host=container.get_container_host_ip(),
                port=int(container.get_exposed_port(5432)),
                user=container.username,
                password=container.password,
                database=container.dbname,
            )
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"Postgres not available: could not start a container ({exc})")


@pytest.fixture(scope="session")
def pg_settings(postgres_params: PostgresParams) -> Settings:
    """Settings pointed at the real database.

    Built through `Settings` rather than by handing a DSN straight to
    SQLAlchemy, so `database_url` and the pool configuration in `create_engine`
    are themselves under test.
    """
    return Settings(
        _env_file=None,
        app_env=AppEnv.DEVELOPMENT,
        jwt_secret_key="test-secret-key-not-used-in-production-0123456789abcdef",
        postgres_host=postgres_params.host,
        postgres_port=postgres_params.port,
        postgres_user=postgres_params.user,
        postgres_password=postgres_params.password,
        postgres_db=postgres_params.database,
        log_format="console",
    )


@pytest.fixture
async def pg_engine(pg_settings: Settings) -> AsyncIterator[AsyncEngine]:
    """An engine against the real database, with the schema ensured.

    Function-scoped, unlike the container. pytest-asyncio gives each test its
    own event loop, and asyncpg connections are bound to the loop that opened
    them -- a session-scoped engine is reused against a loop that has already
    been closed. Creating an engine is cheap; starting Postgres is not, which
    is why only the container is shared.
    """
    engine = create_engine(pg_settings)
    try:
        await create_all(engine)
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def pg_sessions(pg_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory, with the table emptied before and after each test.

    Truncating on entry as well as exit means a test that crashed partway
    through cannot poison the next one.
    """
    factory = create_session_factory(pg_engine)

    async def truncate() -> None:
        async with pg_engine.begin() as connection:
            await connection.execute(text("TRUNCATE TABLE inference_logs RESTART IDENTITY"))

    await truncate()
    try:
        yield factory
    finally:
        await truncate()
