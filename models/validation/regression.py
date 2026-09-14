"""Performance regression detection.

Compares a candidate model's measured accuracy and latency against a recorded
baseline, so a regression is caught before deployment rather than in production.

The problem this solves is that "did it get worse" is not a single question. A
model can regress in accuracy, in latency, or in artefact size, and each has a
different tolerance. A 0.1% accuracy drop is noise; a 5% drop is a release
blocker. A 2ms latency increase is irrelevant; a 200ms increase breaks the SLO.

Tolerances are therefore per-metric and *directional*: an improvement is never
a regression, which sounds obvious but is easy to get wrong with a symmetric
comparison.

Baselines are committed to the repository. A baseline computed on the fly from
recent runs drifts upward with every release — each slightly-worse model becomes
the standard the next one is measured against, and quality erodes without any
single comparison ever failing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Direction(StrEnum):
    """Which way is better for a metric."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """How to compare one metric.

    ``tolerance`` is a relative fraction: 0.02 permits a 2% degradation.
    ``absolute_floor`` optionally sets a hard bound that no amount of relative
    tolerance may breach -- useful for a metric with a contractual minimum.
    """

    name: str
    direction: Direction
    tolerance: float
    absolute_floor: float | None = None

    def regressed(self, baseline: float, candidate: float) -> tuple[bool, str]:
        """Return whether ``candidate`` is unacceptably worse than ``baseline``."""
        if self.direction is Direction.HIGHER_IS_BETTER:
            if self.absolute_floor is not None and candidate < self.absolute_floor:
                return True, (
                    f"{candidate:.4f} is below the absolute floor of {self.absolute_floor:.4f}"
                )
            permitted = baseline * (1 - self.tolerance)
            if candidate < permitted:
                drop = (baseline - candidate) / baseline if baseline else 0.0
                return True, (
                    f"dropped {drop * 100:.2f}% ({baseline:.4f} -> {candidate:.4f}), "
                    f"tolerance is {self.tolerance * 100:.1f}%"
                )
            return False, ""

        if self.absolute_floor is not None and candidate > self.absolute_floor:
            return True, (
                f"{candidate:.2f} exceeds the absolute ceiling of {self.absolute_floor:.2f}"
            )
        permitted = baseline * (1 + self.tolerance)
        if candidate > permitted:
            increase = (candidate - baseline) / baseline if baseline else 0.0
            return True, (
                f"increased {increase * 100:.2f}% ({baseline:.2f} -> {candidate:.2f}), "
                f"tolerance is {self.tolerance * 100:.1f}%"
            )
        return False, ""


#: Metrics gated on release, with the tolerances chosen for this project.
#:
#: Accuracy: 2% relative, plus an absolute floor of 80%. The relative tolerance
#: absorbs run-to-run variance from non-deterministic CUDA kernels; the floor
#: prevents a slow slide across many releases, each individually within
#: tolerance.
#:
#: Latency: 20% relative. Deliberately loose, because CI runners are shared and
#: noisy; a tighter bound produces false failures that trained people to ignore
#: the gate, which is worse than no gate.
DEFAULT_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("acc_top1", Direction.HIGHER_IS_BETTER, tolerance=0.02, absolute_floor=0.80),
    MetricSpec("acc_top5", Direction.HIGHER_IS_BETTER, tolerance=0.02, absolute_floor=0.93),
    MetricSpec("latency_p50_ms", Direction.LOWER_IS_BETTER, tolerance=0.20),
    MetricSpec("latency_p95_ms", Direction.LOWER_IS_BETTER, tolerance=0.25),
    MetricSpec("artifact_size_mb", Direction.LOWER_IS_BETTER, tolerance=0.10),
)


@dataclass(slots=True)
class RegressionResult:
    """Comparison of one metric against its baseline."""

    metric: str
    baseline: float
    candidate: float
    regressed: bool
    detail: str
    improved: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RegressionReport:
    """The full comparison."""

    model_name: str
    baseline_version: str
    candidate_version: str
    results: list[RegressionResult] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the candidate may be released.

        A missing metric fails the check. Silently passing a comparison that
        could not be made would let an unmeasured regression through, which is
        the opposite of what a gate is for.
        """
        return not self.missing_metrics and not any(r.regressed for r in self.results)

    @property
    def regressions(self) -> list[RegressionResult]:
        return [r for r in self.results if r.regressed]

    def summary(self) -> str:
        if self.missing_metrics:
            return f"FAILED: metrics missing from the candidate: {', '.join(self.missing_metrics)}"
        if self.regressions:
            lines = [f"FAILED: {len(self.regressions)} regression(s)"]
            lines.extend(f"  {r.metric}: {r.detail}" for r in self.regressions)
            return "\n".join(lines)

        improved = [r for r in self.results if r.improved]
        suffix = f" ({len(improved)} improved)" if improved else ""
        return f"PASSED: {len(self.results)} metrics within tolerance{suffix}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "passed": self.passed,
            "missing_metrics": self.missing_metrics,
            "results": [r.as_dict() for r in self.results],
            "summary": self.summary(),
        }


def compare_to_baseline(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    specs: tuple[MetricSpec, ...] = DEFAULT_SPECS,
    model_name: str = "",
) -> RegressionReport:
    """Check a candidate's metrics against a recorded baseline."""
    report = RegressionReport(
        model_name=model_name or baseline.get("model_name", ""),
        baseline_version=baseline.get("version", "unknown"),
        candidate_version=candidate.get("version", "candidate"),
    )

    baseline_metrics = baseline.get("metrics", baseline)
    candidate_metrics = candidate.get("metrics", candidate)

    for spec in specs:
        if spec.name not in baseline_metrics:
            # Not in the baseline: a newly tracked metric has nothing to
            # compare against, which is not a failure.
            continue
        if spec.name not in candidate_metrics:
            report.missing_metrics.append(spec.name)
            continue

        baseline_value = float(baseline_metrics[spec.name])
        candidate_value = float(candidate_metrics[spec.name])

        regressed, detail = spec.regressed(baseline_value, candidate_value)

        if spec.direction is Direction.HIGHER_IS_BETTER:
            improved = candidate_value > baseline_value
        else:
            improved = candidate_value < baseline_value

        report.results.append(
            RegressionResult(
                metric=spec.name,
                baseline=baseline_value,
                candidate=candidate_value,
                regressed=regressed,
                improved=improved and not regressed,
                detail=detail
                or (
                    f"{baseline_value:.4f} -> {candidate_value:.4f}"
                    + (" (improved)" if improved else "")
                ),
            )
        )

    logger.info(
        "Regression check for %s: passed=%s, regressions=%s",
        report.model_name,
        report.passed,
        [r.metric for r in report.regressions] or "none",
    )
    return report


def load_baseline(path: Path) -> dict[str, Any]:
    """Read a committed baseline."""
    if not path.exists():
        raise FileNotFoundError(
            f"No baseline at {path}. Record one with `write_baseline` after a "
            f"release you are willing to be measured against."
        )
    return json.loads(path.read_text())


def write_baseline(
    path: Path,
    *,
    model_name: str,
    version: str,
    metrics: dict[str, float],
    notes: str = "",
) -> None:
    """Record a baseline for future comparisons.

    Deliberately a separate, explicit step rather than something a training run
    does automatically. A baseline updated on every run drifts upward with each
    release, so quality erodes without any single comparison ever failing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "version": version,
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "metrics": metrics,
                "notes": notes,
            },
            indent=2,
        )
    )
    logger.info("Wrote baseline for version %s to %s", version, path)
