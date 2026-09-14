#!/usr/bin/env python3
"""Analyse model drift from the inference audit trail.

Compares a recent window of production traffic against an earlier baseline
window, both read from ``inference_logs``.

The comparison is between two *production* windows rather than against the
training set. Training-set statistics describe curated data; what matters
operationally is whether live traffic today looks like live traffic last week,
because that is the change a model actually experiences.

Usage::

    uv run python scripts/analyse_drift.py
    uv run python scripts/analyse_drift.py --baseline-days 30 --current-days 1
    uv run python scripts/analyse_drift.py --model tiny-imagenet-classifier --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger("analyse_drift")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Restrict to one model name")
    parser.add_argument("--version", default=None, help="Restrict to one model version")
    parser.add_argument(
        "--baseline-days",
        type=float,
        default=7.0,
        help="Length of the baseline window, ending where the current window starts",
    )
    parser.add_argument(
        "--current-days", type=float, default=1.0, help="Length of the current window"
    )
    parser.add_argument("--limit", type=int, default=50_000, help="Maximum rows to read per window")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit non-zero when significant drift is detected, for use in a scheduled job",
    )
    parser.add_argument(
        "--push-metrics",
        action="store_true",
        help=(
            "Publish results to the Prometheus pushgateway, so drift alerts route "
            "through the same path as every other alert"
        ),
    )
    parser.add_argument(
        "--pushgateway",
        default="pushgateway:9091",
        help="Prometheus pushgateway address",
    )
    return parser


def push_drift_metrics(report: Any, gateway: str) -> None:
    """Publish drift severity to Prometheus.

    A scheduled job that only writes to stdout is not alerting -- nobody reads
    a container log on a schedule. Pushing to the gateway means drift is
    alertable through exactly the same rules, routing, and on-call rotation as
    every other signal, rather than needing its own notification path.

    The pushgateway (rather than a scrape endpoint) is the right primitive
    here: this is a batch job that exits, so there is nothing for Prometheus to
    scrape.
    """
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    severity_values = {
        "none": 0.0,
        "insufficient_data": 0.0,
        "moderate": 1.0,
        "significant": 2.0,
    }

    registry = CollectorRegistry()

    overall = Gauge(
        "model_drift_severity",
        "Drift severity: 0 none, 1 moderate, 2 significant.",
        ["model", "version"],
        registry=registry,
    )
    overall.labels(
        model=report.model_name or "unknown", version=report.model_version or "unknown"
    ).set(severity_values.get(report.overall_severity.value, 0.0))

    per_feature = Gauge(
        "model_drift_statistic",
        "Per-feature drift statistic (PSI or KS).",
        ["model", "version", "feature", "method"],
        registry=registry,
    )
    for result in report.results:
        per_feature.labels(
            model=report.model_name or "unknown",
            version=report.model_version or "unknown",
            feature=result.feature,
            method=result.method,
        ).set(result.statistic)

    try:
        push_to_gateway(gateway, job="model_drift", registry=registry)
        logger.info("Pushed drift metrics to %s", gateway)
    except Exception as exc:
        # A monitoring failure must not fail the job, or a pushgateway outage
        # looks identical to detected drift.
        logger.warning("Could not push metrics to %s: %s", gateway, exc)


async def fetch_window(
    session_factory: Any, start: datetime, end: datetime, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Read inference records within a time window."""
    from sqlalchemy import text

    # Fully static SQL with bound parameters -- no string assembly, so there is
    # no injection surface to reason about. The `:model IS NULL OR` form makes
    # each filter optional without branching on the query text; the window is
    # already bounded by created_at, which is indexed, so the optional filters
    # cost nothing meaningful for a maintenance query.
    query = text(
        "SELECT top_label, image_format, image_width, image_height, image_bytes, "
        "       latency_ms, status, model_name, model_version "
        "FROM inference_logs "
        "WHERE created_at >= :start "
        "  AND created_at < :end "
        "  AND (CAST(:model AS text) IS NULL OR model_name = :model) "
        "  AND (CAST(:version AS text) IS NULL OR model_version = :version) "
        "ORDER BY created_at DESC "
        "LIMIT :limit"
    )
    params: dict[str, Any] = {
        "start": start,
        "end": end,
        "model": args.model,
        "version": args.version,
        "limit": args.limit,
    }

    async with session_factory() as session:
        result = await session.execute(query, params)
        return [dict(row._mapping) for row in result]


async def run(args: argparse.Namespace) -> int:
    from api.config import get_settings
    from api.db.session import create_engine, create_session_factory
    from models.validation.drift import DriftSeverity, compare_windows

    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    now = datetime.now(UTC)
    current_start = now - timedelta(days=args.current_days)
    baseline_start = current_start - timedelta(days=args.baseline_days)

    try:
        baseline = await fetch_window(session_factory, baseline_start, current_start, args)
        current = await fetch_window(session_factory, current_start, now, args)
    finally:
        await engine.dispose()

    logger.info(
        "Baseline window: %s to %s (%d records)",
        baseline_start.date(),
        current_start.date(),
        len(baseline),
    )
    logger.info(
        "Current window:  %s to %s (%d records)",
        current_start.date(),
        now.date(),
        len(current),
    )

    if not baseline or not current:
        logger.warning(
            "One or both windows are empty. Drift cannot be assessed without "
            "traffic in both periods."
        )
        return 0

    model_name = args.model or baseline[0].get("model_name", "")
    model_version = args.version or baseline[0].get("model_version", "")

    report = compare_windows(baseline, current, model_name=model_name, model_version=model_version)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        _print_table(report)

    if args.push_metrics:
        push_drift_metrics(report, args.pushgateway)

    if args.fail_on_drift and report.overall_severity is DriftSeverity.SIGNIFICANT:
        logger.error("Significant drift detected")
        return 1
    return 0


def _print_table(report: Any) -> None:
    """Render a drift report for a human reader."""
    print(f"\nDrift report: {report.model_name}:{report.model_version}")
    print(f"Overall severity: {report.overall_severity.value.upper()}\n")

    print(f"{'Feature':<18} {'Method':<8} {'Statistic':>10} {'Severity':<18} Detail")
    print("-" * 100)
    for result in report.results:
        print(
            f"{result.feature:<18} {result.method:<8} {result.statistic:>10.4f} "
            f"{result.severity.value:<18} {result.detail[:44]}"
        )

    if report.drifted_features:
        print(
            f"\nDrifted: {', '.join(r.feature for r in report.drifted_features)}\n"
            "Investigate whether the input population changed, then decide "
            "whether the model needs retraining or the change is expected."
        )
    else:
        print("\nNo material drift detected.")


def main(argv: list[str] | None = None) -> int:
    import asyncio

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
