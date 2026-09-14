"""Add monthly partitioning and retention to inference_logs.

Revision ID: b1f4c8a92e01
Revises: 990c1570c7e2
Create Date: 2026-09-14

The audit table receives one row per inference. At sustained production volume
an unpartitioned table becomes a problem in three ways, none of which appear
during development:

1. **Deletes are expensive.** Removing a month of expired rows from a large
   table is a long-running DELETE that bloats the table and competes with
   inserts. Dropping a partition is instant and reclaims space immediately.
2. **Indexes grow without bound.** A single index over years of rows is far
   larger than the working set, so the useful portion no longer fits in cache.
3. **Vacuum falls behind.** Autovacuum on one enormous append-heavy table
   struggles; per-partition vacuum is bounded work.

Partitioning by month on ``created_at`` matches how the data is queried — drift
and A/B analysis both read recent windows — so the planner can prune to one or
two partitions instead of scanning everything.

**This migration is not automatically reversible in the usual sense.** Postgres
cannot convert a partitioned table back to a plain one in place. The downgrade
therefore rebuilds a plain table and copies the data across, which is correct
but slow on a large table. That is stated rather than hidden, because
discovering it during an incident rollback would be considerably worse.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b1f4c8a92e01"
down_revision: str | None = "990c1570c7e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Months of history retained by default. Long enough for
#: year-over-year-adjacent comparison and drift baselines, short enough that
#: the table stays manageable.
RETENTION_MONTHS = 6

#: Token replaced with RETENTION_MONTHS in the SQL below. Substitution rather
#: than an f-string keeps the statements free of interpolated expressions, so
#: neither a reader nor a linter has to verify that nothing untrusted reaches
#: the SQL text.
_RETENTION_MARKER = "RETENTION_MONTHS_PLACEHOLDER"


def _sql(statement: str) -> str:
    """Substitute the retention constant into a fixed SQL template."""
    return statement.replace(_RETENTION_MARKER, str(RETENTION_MONTHS))


def upgrade() -> None:
    # The existing table is renamed rather than dropped, so no data is lost if
    # the copy below fails; the old table remains until it is explicitly
    # dropped at the end.
    op.execute("ALTER TABLE inference_logs RENAME TO inference_logs_unpartitioned")

    # Renaming a table does not rename its indexes, so the old index names are
    # still taken and creating them on the new table would fail with
    # "relation already exists". They are dropped here rather than renamed
    # because the table they belong to is dropped at the end of this migration.
    for index_name in (
        "ix_inference_logs_created_at",
        "ix_inference_logs_model_created",
        "ix_inference_logs_user_created",
        "ix_inference_logs_errors",
        "ix_inference_logs_image_sha256",
        "ix_inference_logs_variant",
    ):
        op.execute(f"DROP INDEX IF EXISTS {index_name}")

    # A partitioned table's primary key must contain the partition key, so the
    # key becomes (id, created_at). This is a real consequence: a lookup by id
    # alone can no longer use the primary key to prune partitions, though it
    # remains correct. Audit rows are queried by time and model, not by id, so
    # the trade is acceptable here.
    op.execute(
        """
        CREATE TABLE inference_logs (
            id                BIGSERIAL       NOT NULL,
            correlation_id    VARCHAR(64)     NOT NULL,
            user_id           VARCHAR(128),
            user_tier         VARCHAR(32),
            variant           VARCHAR(64),
            model_name        VARCHAR(128)    NOT NULL,
            model_version     VARCHAR(64)     NOT NULL,
            backend           VARCHAR(32)     NOT NULL,
            task              VARCHAR(32)     NOT NULL,
            status            VARCHAR(16)     NOT NULL,
            error_code        VARCHAR(64),
            latency_ms        DOUBLE PRECISION NOT NULL,
            cached            BOOLEAN         NOT NULL DEFAULT FALSE,
            batch_size        INTEGER         NOT NULL DEFAULT 1,
            image_sha256      VARCHAR(64),
            image_bytes       INTEGER,
            image_width       INTEGER,
            image_height      INTEGER,
            image_format      VARCHAR(16),
            top_label         VARCHAR(256),
            top_class_id      INTEGER,
            top_probability   DOUBLE PRECISION,
            created_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
            notes             TEXT,
            PRIMARY KEY (id, created_at)
        ) PARTITION BY RANGE (created_at)
        """
    )

    # Indexes declared on the parent are created on every partition, existing
    # and future.
    op.execute("CREATE INDEX ix_inference_logs_created_at ON inference_logs (created_at DESC)")
    op.execute(
        "CREATE INDEX ix_inference_logs_model_created "
        "ON inference_logs (model_name, model_version, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_inference_logs_user_created ON inference_logs (user_id, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_inference_logs_errors ON inference_logs (created_at DESC) "
        "WHERE status <> 'success'"
    )
    op.execute("CREATE INDEX ix_inference_logs_image_sha256 ON inference_logs (image_sha256)")
    op.execute(
        "CREATE INDEX ix_inference_logs_variant ON inference_logs (variant, created_at DESC) "
        "WHERE variant IS NOT NULL"
    )

    # A DEFAULT partition catches rows outside every declared range. Without
    # it an insert with an unexpected timestamp fails outright, which would
    # turn a missing-partition maintenance lapse into failed writes.
    #
    # Two consequences worth knowing. Rows landing here are a signal that
    # partition creation has fallen behind. And `drop_expired_inference_logs`
    # deliberately never drops this partition, because it may hold rows from
    # any period -- so data that arrived while partitions were missing is
    # retained indefinitely and must be cleaned up by hand. Monitor its row
    # count; a non-zero value means the scheduled partition job is not running.
    op.execute("CREATE TABLE inference_logs_default PARTITION OF inference_logs DEFAULT")

    # Create partitions around today so the deployment works immediately.
    # RETENTION_MONTHS is substituted with str.replace on a fixed template
    # rather than an f-string, so no expression is interpolated into SQL text
    # and there is nothing for a reader (or a linter) to have to verify.
    op.execute(
        _sql("""
        DO $$
        DECLARE
            month_start DATE;
            month_end   DATE;
            part_name   TEXT;
        BEGIN
            FOR i IN -RETENTION_MONTHS_PLACEHOLDER..3 LOOP
                month_start := date_trunc('month', CURRENT_DATE + (i || ' month')::interval);
                month_end   := month_start + INTERVAL '1 month';
                part_name   := 'inference_logs_' || to_char(month_start, 'YYYY_MM');

                IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = part_name) THEN
                    EXECUTE format(
                        'CREATE TABLE %I PARTITION OF inference_logs '
                        'FOR VALUES FROM (%L) TO (%L)',
                        part_name, month_start, month_end
                    );
                END IF;
            END LOOP;
        END $$
        """)
    )

    # Copy existing rows. They route to the correct partition automatically.
    op.execute(
        """
        INSERT INTO inference_logs (
            correlation_id, user_id, user_tier, variant, model_name, model_version,
            backend, task, status, error_code, latency_ms, cached, batch_size,
            image_sha256, image_bytes, image_width, image_height, image_format,
            top_label, top_class_id, top_probability, created_at, notes
        )
        SELECT
            correlation_id, user_id, user_tier, variant, model_name, model_version,
            backend, task, status, error_code, latency_ms, cached, batch_size,
            image_sha256, image_bytes, image_width, image_height, image_format,
            top_label, top_class_id, top_probability, created_at, notes
        FROM inference_logs_unpartitioned
        """
    )

    op.execute("DROP TABLE inference_logs_unpartitioned")

    # Maintenance helpers. Kept in the database rather than application code so
    # they can be run by a scheduled job, a DBA, or psql during an incident,
    # without deploying anything.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION create_inference_log_partitions(months_ahead INT DEFAULT 3)
        RETURNS TABLE(created TEXT) AS $$
        DECLARE
            month_start DATE;
            month_end   DATE;
            part_name   TEXT;
        BEGIN
            FOR i IN 0..months_ahead LOOP
                month_start := date_trunc('month', CURRENT_DATE + (i || ' month')::interval);
                month_end   := month_start + INTERVAL '1 month';
                part_name   := 'inference_logs_' || to_char(month_start, 'YYYY_MM');

                IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = part_name) THEN
                    EXECUTE format(
                        'CREATE TABLE %I PARTITION OF inference_logs '
                        'FOR VALUES FROM (%L) TO (%L)',
                        part_name, month_start, month_end
                    );
                    created := part_name;
                    RETURN NEXT;
                END IF;
            END LOOP;
        END $$ LANGUAGE plpgsql
        """
    )

    op.execute(
        _sql("""
        CREATE OR REPLACE FUNCTION drop_expired_inference_logs(
            retain_months INT DEFAULT RETENTION_MONTHS_PLACEHOLDER
        )
        RETURNS TABLE(dropped TEXT) AS $$
        DECLARE
            cutoff    DATE;
            part      RECORD;
            part_date DATE;
        BEGIN
            cutoff := date_trunc('month', CURRENT_DATE - (retain_months || ' month')::interval);

            FOR part IN
                SELECT c.relname
                FROM pg_class c
                JOIN pg_inherits i ON i.inhrelid = c.oid
                JOIN pg_class p ON p.oid = i.inhparent
                WHERE p.relname = 'inference_logs'
                  AND c.relname <> 'inference_logs_default'
            LOOP
                -- The date is derived from the partition name, which this
                -- migration controls, rather than by parsing pg_get_expr. The
                -- printed bound is a timestamptz carrying a timezone suffix
                -- ('2026-03-01 00:00:00+00'), so a regex expecting a bare date
                -- matches nothing -- and a failed match here is indistinguishable
                -- from "nothing is expired", so retention would silently never
                -- run.
                part_date := to_date(right(part.relname, 7), 'YYYY_MM');

                IF part_date < cutoff THEN
                    -- DROP, not DELETE: instant, and reclaims space at once.
                    EXECUTE format('DROP TABLE %I', part.relname);
                    dropped := part.relname;
                    RETURN NEXT;
                END IF;
            END LOOP;
        END $$ LANGUAGE plpgsql
        """)
    )


def downgrade() -> None:
    """Rebuild a plain table.

    Postgres cannot un-partition in place, so this copies every row into a new
    table. On a large table that is slow and needs the disk space for both
    copies simultaneously. Plan a rollback accordingly.
    """
    op.execute("DROP FUNCTION IF EXISTS drop_expired_inference_logs(INT)")
    op.execute("DROP FUNCTION IF EXISTS create_inference_log_partitions(INT)")

    op.execute("ALTER TABLE inference_logs RENAME TO inference_logs_partitioned")

    op.execute(
        """
        CREATE TABLE inference_logs (
            id                SERIAL          PRIMARY KEY,
            correlation_id    VARCHAR(64)     NOT NULL,
            user_id           VARCHAR(128),
            user_tier         VARCHAR(32),
            variant           VARCHAR(64),
            model_name        VARCHAR(128)    NOT NULL,
            model_version     VARCHAR(64)     NOT NULL,
            backend           VARCHAR(32)     NOT NULL,
            task              VARCHAR(32)     NOT NULL,
            status            VARCHAR(16)     NOT NULL,
            error_code        VARCHAR(64),
            latency_ms        DOUBLE PRECISION NOT NULL,
            cached            BOOLEAN         NOT NULL DEFAULT FALSE,
            batch_size        INTEGER         NOT NULL DEFAULT 1,
            image_sha256      VARCHAR(64),
            image_bytes       INTEGER,
            image_width       INTEGER,
            image_height      INTEGER,
            image_format      VARCHAR(16),
            top_label         VARCHAR(256),
            top_class_id      INTEGER,
            top_probability   DOUBLE PRECISION,
            created_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
            notes             TEXT
        )
        """
    )

    op.execute(
        """
        INSERT INTO inference_logs (
            correlation_id, user_id, user_tier, variant, model_name, model_version,
            backend, task, status, error_code, latency_ms, cached, batch_size,
            image_sha256, image_bytes, image_width, image_height, image_format,
            top_label, top_class_id, top_probability, created_at, notes
        )
        SELECT
            correlation_id, user_id, user_tier, variant, model_name, model_version,
            backend, task, status, error_code, latency_ms, cached, batch_size,
            image_sha256, image_bytes, image_width, image_height, image_format,
            top_label, top_class_id, top_probability, created_at, notes
        FROM inference_logs_partitioned
        """
    )

    op.execute("DROP TABLE inference_logs_partitioned CASCADE")

    op.create_index("ix_inference_logs_created_at", "inference_logs", [sa.text("created_at DESC")])
    op.create_index(
        "ix_inference_logs_model_created",
        "inference_logs",
        ["model_name", "model_version", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_inference_logs_user_created",
        "inference_logs",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_inference_logs_errors",
        "inference_logs",
        [sa.text("created_at DESC")],
        postgresql_where=sa.text("status <> 'success'"),
    )
    op.create_index("ix_inference_logs_image_sha256", "inference_logs", ["image_sha256"])
    op.create_index(
        "ix_inference_logs_variant",
        "inference_logs",
        ["variant", sa.text("created_at DESC")],
        postgresql_where=sa.text("variant IS NOT NULL"),
    )
