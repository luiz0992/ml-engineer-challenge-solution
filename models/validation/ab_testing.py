"""A/B testing across model versions.

Splits traffic between a control and one or more challenger versions, then
compares outcomes from the ``inference_logs`` audit trail.

Design decisions that matter:

**Assignment is deterministic per user, not random per request.** A user hashed
to the challenger stays on the challenger. Random per-request assignment would
give the same caller different model versions for identical inputs, which is
both confusing and statistically wrong — the observations within a user are then
correlated across arms, violating the independence a significance test assumes.

**Hashing, not a stored assignment table.** ``hash(user_id + experiment)``
requires no storage, is stable across restarts and replicas, and gives a
different split per experiment so a user unlucky in one is not systematically in
every challenger arm.

**The comparison reports effect size and confidence intervals, not just a
p-value.** At production volumes any difference becomes statistically
significant; what matters is whether it is *large enough to act on*. A 0.2%
latency regression with p < 0.001 is not a reason to abandon a rollout.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Minimum observations per arm before a comparison is reported. Below this the
#: confidence interval is wider than any effect worth detecting.
MIN_ARM_SAMPLES = 100


@dataclass(frozen=True, slots=True)
class Variant:
    """One arm of an experiment."""

    name: str
    model_version: str
    #: Share of traffic, between 0 and 1.
    weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")


@dataclass(slots=True)
class Experiment:
    """A traffic-splitting experiment across model versions."""

    name: str
    variants: list[Variant]
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError("An experiment needs at least one variant")

        total = sum(v.weight for v in self.variants)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Variant weights must sum to 1.0, got {total}")

        names = [v.name for v in self.variants]
        if len(names) != len(set(names)):
            raise ValueError(f"Variant names must be unique, got {names}")

    def assign(self, user_id: str) -> Variant:
        """Assign a user to a variant, deterministically and stably.

        The experiment name is part of the hash input so a user's position in
        one experiment does not determine their position in another; without it
        the same users would land in the challenger arm of every experiment and
        results across experiments would be correlated.
        """
        if not self.enabled:
            return self.variants[0]

        digest = hashlib.sha256(f"{self.name}:{user_id}".encode()).digest()
        # First 8 bytes as an integer, mapped to [0, 1).
        position = int.from_bytes(digest[:8], "big") / (2**64)

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if position < cumulative:
                return variant

        # Floating-point accumulation can leave `position` marginally above the
        # final cumulative bound; the last variant is the correct fallback.
        return self.variants[-1]

    @property
    def control(self) -> Variant:
        return self.variants[0]


@dataclass(slots=True)
class ArmMetrics:
    """Observed outcomes for one arm."""

    variant: str
    model_version: str
    count: int
    success_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "model_version": self.model_version,
            "count": self.count,
            "success_rate": round(self.success_rate, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "mean_confidence": (
                round(self.mean_confidence, 4) if self.mean_confidence is not None else None
            ),
        }


@dataclass(slots=True)
class Comparison:
    """A control-versus-challenger comparison of one metric."""

    metric: str
    control_value: float
    treatment_value: float
    absolute_difference: float
    relative_difference: float
    confidence_interval: tuple[float, float]
    p_value: float
    significant: bool
    #: Whether the difference is large enough to act on, independent of p.
    material: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "control": round(self.control_value, 4),
            "treatment": round(self.treatment_value, 4),
            "absolute_difference": round(self.absolute_difference, 4),
            "relative_difference_pct": round(self.relative_difference * 100, 2),
            "confidence_interval_95": [
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4),
            ],
            "p_value": round(self.p_value, 6),
            "statistically_significant": self.significant,
            "materially_different": self.material,
        }


@dataclass(slots=True)
class ExperimentReport:
    """Results of an experiment."""

    experiment: str
    arms: list[ArmMetrics] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    recommendation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "arms": [a.as_dict() for a in self.arms],
            "comparisons": [c.as_dict() for c in self.comparisons],
            "recommendation": self.recommendation,
        }


def summarise_arm(
    variant: str, model_version: str, records: Sequence[dict[str, Any]]
) -> ArmMetrics:
    """Reduce an arm's inference records to comparable metrics."""
    latencies = np.array([r["latency_ms"] for r in records], dtype=float)
    successes = sum(1 for r in records if r.get("status") == "success")

    confidences = [r["top_probability"] for r in records if r.get("top_probability") is not None]

    return ArmMetrics(
        variant=variant,
        model_version=model_version,
        count=len(records),
        success_rate=successes / len(records) if records else 0.0,
        mean_latency_ms=float(latencies.mean()) if latencies.size else 0.0,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies.size else 0.0,
        mean_confidence=float(np.mean(confidences)) if confidences else None,
    )


def compare_means(
    metric: str,
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    material_threshold: float = 0.05,
) -> Comparison:
    """Compare two arms on a continuous metric.

    Uses Welch's t-test rather than Student's, because the arms have no reason
    to share a variance — a slower backend is usually also more variable, and
    assuming equal variance would understate the uncertainty.

    ``material_threshold`` is a *relative* difference: a change smaller than
    this is reported as immaterial however significant it is. This is the guard
    against the production failure mode where every comparison is significant
    because n is large.
    """
    from scipy import stats

    control_array = np.asarray(control, dtype=float)
    treatment_array = np.asarray(treatment, dtype=float)

    control_mean = float(control_array.mean())
    treatment_mean = float(treatment_array.mean())
    absolute = treatment_mean - control_mean
    relative = absolute / control_mean if control_mean else 0.0

    result = stats.ttest_ind(treatment_array, control_array, equal_var=False)

    # Welch-Satterthwaite standard error for the difference of means.
    standard_error = math.sqrt(
        control_array.var(ddof=1) / control_array.size
        + treatment_array.var(ddof=1) / treatment_array.size
    )
    margin = 1.96 * standard_error

    return Comparison(
        metric=metric,
        control_value=control_mean,
        treatment_value=treatment_mean,
        absolute_difference=absolute,
        relative_difference=relative,
        confidence_interval=(absolute - margin, absolute + margin),
        p_value=float(result.pvalue),
        significant=bool(result.pvalue < 0.05),
        material=abs(relative) >= material_threshold,
    )


def compare_proportions(
    metric: str,
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    *,
    material_threshold: float = 0.01,
) -> Comparison:
    """Compare two arms on a rate, such as success rate.

    Uses a two-proportion z-test with a pooled estimate under the null.
    """
    from scipy import stats

    control_rate = control_successes / control_total if control_total else 0.0
    treatment_rate = treatment_successes / treatment_total if treatment_total else 0.0
    absolute = treatment_rate - control_rate
    relative = absolute / control_rate if control_rate else 0.0

    pooled = (control_successes + treatment_successes) / max(control_total + treatment_total, 1)
    pooled_se = math.sqrt(
        pooled * (1 - pooled) * (1 / max(control_total, 1) + 1 / max(treatment_total, 1))
    )

    z = absolute / pooled_se if pooled_se else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    # The interval uses unpooled standard errors: pooling is appropriate for
    # the test statistic under the null, but not for estimating the interval
    # around an observed difference.
    unpooled_se = math.sqrt(
        control_rate * (1 - control_rate) / max(control_total, 1)
        + treatment_rate * (1 - treatment_rate) / max(treatment_total, 1)
    )
    margin = 1.96 * unpooled_se

    return Comparison(
        metric=metric,
        control_value=control_rate,
        treatment_value=treatment_rate,
        absolute_difference=absolute,
        relative_difference=relative,
        confidence_interval=(absolute - margin, absolute + margin),
        p_value=p_value,
        significant=bool(p_value < 0.05),
        material=abs(absolute) >= material_threshold,
    )


def analyse_experiment(
    experiment: Experiment,
    records_by_variant: dict[str, list[dict[str, Any]]],
) -> ExperimentReport:
    """Compare every challenger against the control arm."""
    report = ExperimentReport(experiment=experiment.name)

    for variant in experiment.variants:
        records = records_by_variant.get(variant.name, [])
        if records:
            report.arms.append(summarise_arm(variant.name, variant.model_version, records))

    control_records = records_by_variant.get(experiment.control.name, [])
    if len(control_records) < MIN_ARM_SAMPLES:
        report.recommendation = (
            f"Insufficient data: the control arm has {len(control_records)} "
            f"observations, below the {MIN_ARM_SAMPLES} required."
        )
        return report

    for variant in experiment.variants[1:]:
        treatment_records = records_by_variant.get(variant.name, [])
        if len(treatment_records) < MIN_ARM_SAMPLES:
            continue

        report.comparisons.append(
            compare_means(
                f"{variant.name}.latency_ms",
                [r["latency_ms"] for r in control_records],
                [r["latency_ms"] for r in treatment_records],
            )
        )
        report.comparisons.append(
            compare_proportions(
                f"{variant.name}.success_rate",
                sum(1 for r in control_records if r.get("status") == "success"),
                len(control_records),
                sum(1 for r in treatment_records if r.get("status") == "success"),
                len(treatment_records),
            )
        )

    report.recommendation = _recommend(report)
    logger.info(
        "Experiment %s analysed: %d arm(s) -- %s",
        experiment.name,
        len(report.arms),
        report.recommendation,
    )
    return report


def _recommend(report: ExperimentReport) -> str:
    """Turn comparisons into a recommendation.

    A change must be both statistically significant *and* materially large
    before it is called a regression. Significance alone, at production volume,
    would block every rollout.
    """
    if not report.comparisons:
        return "Insufficient data in one or more challenger arms."

    regressions = [
        c
        for c in report.comparisons
        if c.significant
        and c.material
        and (
            (c.metric.endswith("latency_ms") and c.absolute_difference > 0)
            or (c.metric.endswith("success_rate") and c.absolute_difference < 0)
        )
    ]
    if regressions:
        names = ", ".join(c.metric for c in regressions)
        return f"Do not promote: significant and material regression in {names}."

    improvements = [c for c in report.comparisons if c.significant and c.material]
    if improvements:
        return "Safe to promote: significant and material improvement, no regressions."

    return (
        "No material difference detected. Either arm is acceptable; prefer the "
        "simpler or cheaper one."
    )
