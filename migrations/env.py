"""Alembic environment.

Two deliberate choices:

**The database URL comes from application settings**, not ``alembic.ini``.
Credentials then live in exactly one place, and a migration cannot be run
against a different database than the application uses — a mistake that is
easy to make and expensive to discover.

**The synchronous driver is used.** Alembic's migration context is synchronous,
and driving an async engine through it adds a greenlet shim for no benefit;
migrations are a short-lived administrative task where concurrency is
irrelevant.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from api.config import get_settings
from api.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate compares the live schema against this metadata.
target_metadata = Base.metadata

config.set_main_option("sqlalchemy.url", get_settings().sync_database_url)


def run_migrations_offline() -> None:
    """Emit SQL without connecting.

    Used to produce a script for review or for a DBA to apply by hand, which is
    how schema changes reach production in many organisations.
    """
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Detect column type changes, which Alembic ignores by default and
            # which are a common source of silent schema drift.
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
