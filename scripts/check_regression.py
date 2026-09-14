#!/usr/bin/env python3
"""Check a candidate model against the committed performance baseline.

Intended to run in CI before a model artefact is promoted. Exits non-zero on a
regression, so a degraded model cannot be released silently.

Usage::

    uv run python scripts/check_regression.py
    uv run python scripts/check_regression.py --candidate models/artifacts/runs/<run>
    uv run python scripts/check_regression.py --update-baseline   # after a deliberate change
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("check_regression")

DEFAULT_BASELINE = Path("models/validation/baselines/classifier.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, help="Training run to check")
    parser.add_argument("--runs-dir", type=Path, default=Path("models/artifacts/runs"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "Overwrite the baseline with the candidate's metrics. Only after a "
            "deliberate, reviewed change -- an automatic update would let quality "
            "erode one acceptable step at a time."
        ),
    )
    return parser


def collect_candidate_metrics(run_dir: Path, artifacts_dir: Path) -> dict[str, float]:
    """Gather the gated metrics for a training run.

    Accuracy comes from the run's own metrics file. Latency comes from the
    benchmark report if one exists; it is omitted rather than invented when
    absent, and the gate then reports it as missing rather than passing a
    comparison it could not make.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"No metrics.json in {run_dir}")

    run_metrics = json.loads(metrics_path.read_text())
    metrics: dict[str, float] = {}

    for source, target in (("final_acc_top1", "acc_top1"), ("final_acc_top5", "acc_top5")):
        if source in run_metrics:
            metrics[target] = float(run_metrics[source])

    classifier = artifacts_dir / "onnx" / "classifier_fp32.onnx"
    if classifier.exists():
        metrics["artifact_size_mb"] = round(classifier.stat().st_size / 1e6, 1)

    benchmark = Path("benchmarks/results.json")
    if benchmark.exists():
        results = json.loads(benchmark.read_text())["results"]
        serving = [
            r for r in results if r["backend"] == "onnxruntime-cuda" and r["batch_size"] == 1
        ]
        if serving:
            metrics["latency_p50_ms"] = round(serving[0]["p50_ms"], 2)
            metrics["latency_p95_ms"] = round(serving[0]["p95_ms"], 2)
    else:
        logger.warning(
            "No benchmarks/results.json; latency will be reported as missing. "
            "Run `python -m models.optimisation.run_benchmarks` first."
        )

    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    from models.validation.regression import (
        compare_to_baseline,
        load_baseline,
        write_baseline,
    )
    from scripts.prepare_artifacts import find_latest_run

    run_dir = args.candidate or find_latest_run(args.runs_dir)
    candidate_metrics = collect_candidate_metrics(run_dir, args.artifacts_dir)

    logger.info("Candidate: %s", run_dir)
    for name, value in sorted(candidate_metrics.items()):
        logger.info("  %-20s %s", name, value)

    if args.update_baseline:
        write_baseline(
            args.baseline,
            model_name="tiny-imagenet-classifier",
            version=run_dir.name,
            metrics=candidate_metrics,
            notes=f"Updated from {run_dir}",
        )
        logger.warning(
            "Baseline overwritten. Future candidates are now measured against these numbers."
        )
        return 0

    baseline = load_baseline(args.baseline)
    report = compare_to_baseline(baseline, {"version": run_dir.name, "metrics": candidate_metrics})

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        _print_table(report)

    return 0 if report.passed else 1


def _print_table(report: Any) -> None:
    print(
        f"\nRegression check: {report.model_name} "
        f"({report.baseline_version} -> {report.candidate_version})\n"
    )
    print(f"{'Metric':<20} {'Baseline':>12} {'Candidate':>12}  Status")
    print("-" * 78)
    for result in report.results:
        status = "REGRESSED" if result.regressed else ("improved" if result.improved else "ok")
        print(f"{result.metric:<20} {result.baseline:>12.4f} {result.candidate:>12.4f}  {status}")
        if result.regressed:
            print(f"{'':<20} {result.detail}")

    print(f"\n{report.summary()}")


if __name__ == "__main__":
    sys.exit(main())
