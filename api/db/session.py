"""Database engine and session management.

The engine is created once at startup and disposed at shutdown. Sessions are
short-lived and scoped to a unit of work.

Pool sizing is explicit rather than left at SQLAlchemy's defaults. Each API
replica holds up to ``pool_size + max_overflow`` connections, and Postgres has a
finite ``max_connections``; the default of 5 + 10 per process silently becomes
45 across three replicas, which is fine, but the arithmetic has to be
deliberate rather than discovered during an incident.

``pool_pre_ping`` is enabled because connections idle in the pool are routinely
killed by the database, a proxy, or a network device. Without it the first
request after an idle period fails with a stale-connection error that looks like
a database outage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from api.config import Settings
from api.logging_config import get_logger

logger = get_logger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    """Build the async engine for this process."""
    return create_async_engine(
        settings.database_url,
        # Verify a pooled connection before handing it out. Costs one round
        # trip; prevents the stale-connection failure described above.
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        # Recycle below typical proxy and database idle timeouts, so we close
        # connections before something else does it for us.
        pool_recycle=1800,
        # Fail fast rather than queueing behind an exhausted pool: a request
        # that waits 30 seconds for a connection has already missed its budget.
        pool_timeout=5,
        echo=False,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build the session factory.

    ``expire_on_commit=False`` so ORM objects remain usable after commit. The
    default expires every attribute, and the next access would emit a lazy
    reload against a closed session — which in async SQLAlchemy raises rather
    than silently working.
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Provide a transactional session, committing or rolling back."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_connection(engine: AsyncEngine) -> tuple[bool, str | None]:
    """Probe connectivity for the health endpoint."""
    from sqlalchemy import text

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


async def create_all(engine: AsyncEngine) -> None:
    """Create tables directly from the models.

    A convenience for tests and local development only. Production schema
    changes go through Alembic, which is versioned, reviewable, and reversible;
    ``create_all`` cannot alter an existing table and would silently do nothing
    against a schema that has drifted.
    """
    from api.db.models import Base

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    logger.info("database_schema_created", tables=list(Base.metadata.tables))
