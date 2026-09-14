#!/usr/bin/env python3
"""Maintain the audit table's partitions.

Creates upcoming monthly partitions and drops expired ones, then reports the
DEFAULT partition's row count to Prometheus.

Extracted from an inline shell command in the Compose file. Inline scripts
cannot be tested, linted, or type-checked, and this one had grown to the point
where a syntax error would only surface at 2am on the day it next ran.

The partition job must stay *ahead* of inserts. Rows arriving with no matching
partition land in the DEFAULT partition, which retention deliberately never
drops — so falling behind quietly accumulates data that nothing will clean up.

Usage::

    uv run python scripts/maintain_partitions.py
    uv run python scripts/maintain_partitions.py --months-ahead 6 --retain-months 12
    uv run python scripts/maintain_partitions.py --push-metrics
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

logger = logging.getLogger("maintain_partitions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--months-ahead",
        type=int,
        default=3,
        help="Months of future partitions to ensure exist",
    )
    parser.add_argument(
        "--retain-months",
        type=int,
        default=6,
        help="Months of history to keep; older partitions are dropped",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without altering anything",
    )
    parser.add_argument("--push-metrics", action="store_true")
    parser.add_argument("--pushgateway", default="pushgateway:9091")
    return parser


async def maintain(args: argparse.Namespace) -> dict[str, Any]:
    """Run partition maintenance and return what changed."""
    from sqlalchemy import text

    from api.config import get_settings
    from api.db.session import create_engine

    engine = create_engine(get_settings())

    try:
        async with engine.begin() as connection:
            if args.dry_run:
                # Report the current state without creating or dropping.
                existing = (
                    await connection.execute(
                        text(
                            "SELECT c.relname FROM pg_class c "
                            "JOIN pg_inherits i ON i.inhrelid = c.oid "
                            "JOIN pg_class p ON p.oid = i.inhparent "
                            "WHERE p.relname = 'inference_logs' ORDER BY c.relname"
                        )
                    )
                ).fetchall()
                created: list[str] = []
                dropped: list[str] = []
                logger.info("Dry run: %d partitions exist", len(existing))
                for row in existing:
                    logger.info("  %s", row[0])
            else:
                created = [
                    row[0]
                    for row in (
                        await connection.execute(
                            text("SELECT * FROM create_inference_log_partitions(:n)"),
                            {"n": args.months_ahead},
                        )
                    ).fetchall()
                ]
                dropped = [
                    row[0]
                    for row in (
                        await connection.execute(
                            text("SELECT * FROM drop_expired_inference_logs(:n)"),
                            {"n": args.retain_months},
                        )
                    ).fetchall()
                ]

            default_rows = (
                await connection.execute(text("SELECT count(*) FROM inference_logs_default"))
            ).scalar() or 0
    finally:
        await engine.dispose()

    return {"created": created, "dropped": dropped, "default_partition_rows": int(default_rows)}


def push_metrics(result: dict[str, Any], gateway: str) -> None:
    """Publish the DEFAULT partition row count.

    The ``AuditDefaultPartitionNonEmpty`` alert reads this metric. Without the
    push the alert sits permanently green — which is worse than having no
    alert, because it looks like coverage and stops anyone asking whether the
    condition is monitored.
    """
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()

    Gauge(
        "inference_logs_default_partition_rows",
        "Audit rows that fell outside every declared partition range.",
        registry=registry,
    ).set(result["default_partition_rows"])

    Gauge(
        "inference_logs_partitions_created",
        "Partitions created by the most recent maintenance run.",
        registry=registry,
    ).set(len(result["created"]))

    Gauge(
        "inference_logs_partitions_dropped",
        "Partitions dropped by the most recent maintenance run.",
        registry=registry,
    ).set(len(result["dropped"]))

    try:
        push_to_gateway(gateway, job="audit_maintenance", registry=registry)
        logger.info("Pushed metrics to %s", gateway)
    except Exception as exc:
        # A monitoring failure must not fail maintenance, or a pushgateway
        # outage looks identical to a failed partition job.
        logger.warning("Could not push metrics to %s: %s", gateway, exc)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    result = asyncio.run(maintain(args))

    logger.info(
        "created=%s dropped=%s",
        result["created"] or "none",
        result["dropped"] or "none",
    )

    if args.push_metrics:
        push_metrics(result, args.pushgateway)

    if result["default_partition_rows"]:
        logger.warning(
            "%d rows are in the DEFAULT partition. Partition creation has fallen "
            "behind, and retention will never drop them -- they need manual "
            "cleanup once the correct partitions exist.",
            result["default_partition_rows"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
