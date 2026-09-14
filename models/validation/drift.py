"""Model drift detection.

Detects when live traffic has moved away from the distribution a model was
validated on. Two complementary signals are computed from the ``inference_logs``
audit trail:

* **Data drift** — the inputs changed. Image dimensions, formats, and sizes shift
  when a client changes camera, resolution, or upload pipeline.
* **Prediction drift** — the outputs changed. The distribution over predicted
  classes shifts even when inputs look superficially similar.

Prediction drift is the more practical signal here, because it needs no labels.
Ground truth for production traffic usually arrives late or never, so accuracy
cannot be monitored directly; a sustained change in *what the model predicts* is
the earliest available warning that something has moved.

Two statistics are used, deliberately:

**Kolmogorov-Smirnov** for continuous variables (image dimensions, latency). It
is distribution-free and needs no binning, but its p-value is sensitive to
sample size: with 100,000 requests a statistically significant difference is
almost guaranteed and practically meaningless. The KS *statistic* is therefore
reported and thresholded alongside the p-value, never the p-value alone.

**Population Stability Index** for categorical variables (predicted class,
image format). PSI is the industry convention for exactly this, has
well-established interpretation bands, and is insensitive to sample size in the
way a hypothesis test is not.

    PSI < 0.1   no material shift
    0.1 - 0.25  moderate shift, investigate
    PSI > 0.25  significant shift, act

Those bands come from credit-risk practice. They are conventions, not theory,
and are reported as such.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

#: Anything these functions can consume: a Python sequence, a numpy array, or a
#: pandas column. They all call np.asarray internally, so annotating the
#: narrower Sequence[float] misrepresents the API -- a numpy array does not
#: satisfy Sequence structurally, and every caller passing one was a type error
#: that nothing checked.
Numeric = Sequence[float] | npt.NDArray[np.floating]
Categorical = Sequence[Any] | npt.NDArray[Any]

#: Conventional PSI interpretation bands.
PSI_NO_SHIFT = 0.10
PSI_MODERATE_SHIFT = 0.25

#: KS statistic above which a continuous shift is treated as material,
#: independent of the p-value.
KS_MATERIAL_STATISTIC = 0.15

#: Below this, an estimate is too noisy to act on regardless of what the
#: statistic says.
MIN_SAMPLES = 100


class DriftSeverity(StrEnum):
    """How much a distribution has moved."""

    NONE = "none"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(slots=True)
class DriftResult:
    """Outcome for one monitored feature."""

    feature: str
    statistic: float
    method: str
    severity: DriftSeverity
    p_value: float | None = None
    baseline_size: int = 0
    current_size: int = 0
    detail: str = ""

    @property
    def drifted(self) -> bool:
        return self.severity in (DriftSeverity.MODERATE, DriftSeverity.SIGNIFICANT)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "method": self.method,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "severity": self.severity.value,
            "drifted": self.drifted,
            "baseline_size": self.baseline_size,
            "current_size": self.current_size,
            "detail": self.detail,
        }


@dataclass(slots=True)
class DriftReport:
    """Drift across every monitored feature."""

    results: list[DriftResult] = field(default_factory=list)
    model_name: str = ""
    model_version: str = ""

    @property
    def drifted_features(self) -> list[DriftResult]:
        return [r for r in self.results if r.drifted]

    @property
    def overall_severity(self) -> DriftSeverity:
        """The worst severity across features.

        Any single significantly drifted feature makes the report significant:
        drift in one input dimension is enough to invalidate a model, and
        averaging would dilute exactly the signal worth acting on.
        """
        if not self.results:
            return DriftSeverity.INSUFFICIENT_DATA
        if any(r.severity is DriftSeverity.SIGNIFICANT for r in self.results):
            return DriftSeverity.SIGNIFICANT
        if any(r.severity is DriftSeverity.MODERATE for r in self.results):
            return DriftSeverity.MODERATE
        if all(r.severity is DriftSeverity.INSUFFICIENT_DATA for r in self.results):
            return DriftSeverity.INSUFFICIENT_DATA
        return DriftSeverity.NONE

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "overall_severity": self.overall_severity.value,
            "drifted_features": [r.feature for r in self.drifted_features],
            "results": [r.as_dict() for r in self.results],
        }


def population_stability_index(
    baseline: Categorical,
    current: Categorical,
    *,
    epsilon: float = 1e-6,
) -> tuple[float, dict[str, float]]:
    """Compute PSI between two categorical distributions.

    ``PSI = sum((current_i - baseline_i) * ln(current_i / baseline_i))``

    ``epsilon`` replaces zero proportions. Without it a category present in one
    period and absent in the other produces ``ln(0)`` or division by zero, and
    the whole statistic becomes infinite — which is precisely the case that
    matters most, since a newly appearing or vanishing class is strong evidence
    of drift. Substituting a small constant keeps the contribution large but
    finite.

    Returns the PSI and the per-category contributions, so a caller can see
    *which* categories moved rather than only that something did.
    """
    baseline_counts = Counter(baseline)
    current_counts = Counter(current)

    categories = set(baseline_counts) | set(current_counts)
    baseline_total = max(len(baseline), 1)
    current_total = max(len(current), 1)

    psi = 0.0
    contributions: dict[str, float] = {}

    for category in categories:
        baseline_share = max(baseline_counts.get(category, 0) / baseline_total, epsilon)
        current_share = max(current_counts.get(category, 0) / current_total, epsilon)

        contribution = (current_share - baseline_share) * math.log(current_share / baseline_share)
        psi += contribution
        contributions[str(category)] = contribution

    return psi, contributions


def kolmogorov_smirnov(baseline: Numeric, current: Numeric) -> tuple[float, float]:
    """Two-sample KS test. Returns ``(statistic, p_value)``.

    The statistic is the maximum absolute difference between the two empirical
    cumulative distributions, and is the value to act on; the p-value is
    reported for completeness but is dominated by sample size at production
    volumes.
    """
    from scipy import stats

    result = stats.ks_2samp(np.asarray(baseline, dtype=float), np.asarray(current, dtype=float))
    return float(result.statistic), float(result.pvalue)


def detect_categorical_drift(
    feature: str,
    baseline: Categorical,
    current: Categorical,
) -> DriftResult:
    """Assess drift in a categorical feature using PSI."""
    if len(baseline) < MIN_SAMPLES or len(current) < MIN_SAMPLES:
        return DriftResult(
            feature=feature,
            statistic=0.0,
            method="psi",
            severity=DriftSeverity.INSUFFICIENT_DATA,
            baseline_size=len(baseline),
            current_size=len(current),
            detail=f"Need at least {MIN_SAMPLES} samples in each period.",
        )

    psi, contributions = population_stability_index(baseline, current)

    if psi >= PSI_MODERATE_SHIFT:
        severity = DriftSeverity.SIGNIFICANT
    elif psi >= PSI_NO_SHIFT:
        severity = DriftSeverity.MODERATE
    else:
        severity = DriftSeverity.NONE

    # Name the biggest movers, so a report is actionable rather than a number.
    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    detail = ", ".join(f"{name} ({value:+.3f})" for name, value in top)

    return DriftResult(
        feature=feature,
        statistic=psi,
        method="psi",
        severity=severity,
        baseline_size=len(baseline),
        current_size=len(current),
        detail=f"Largest contributors: {detail}" if top else "",
    )


def detect_continuous_drift(
    feature: str,
    baseline: Numeric,
    current: Numeric,
) -> DriftResult:
    """Assess drift in a continuous feature using the KS test.

    Severity is driven by the KS statistic rather than the p-value. At
    production volumes a hypothesis test rejects the null for differences far
    too small to matter, so a p-value alone would report permanent drift.
    """
    if len(baseline) < MIN_SAMPLES or len(current) < MIN_SAMPLES:
        return DriftResult(
            feature=feature,
            statistic=0.0,
            method="ks",
            severity=DriftSeverity.INSUFFICIENT_DATA,
            baseline_size=len(baseline),
            current_size=len(current),
            detail=f"Need at least {MIN_SAMPLES} samples in each period.",
        )

    statistic, p_value = kolmogorov_smirnov(baseline, current)

    if statistic >= KS_MATERIAL_STATISTIC * 2:
        severity = DriftSeverity.SIGNIFICANT
    elif statistic >= KS_MATERIAL_STATISTIC:
        severity = DriftSeverity.MODERATE
    else:
        severity = DriftSeverity.NONE

    baseline_median = float(np.median(baseline))
    current_median = float(np.median(current))

    return DriftResult(
        feature=feature,
        statistic=statistic,
        method="ks",
        severity=severity,
        p_value=p_value,
        baseline_size=len(baseline),
        current_size=len(current),
        detail=(
            f"median {baseline_median:.1f} -> {current_median:.1f}; "
            f"p={p_value:.2e} (statistic drives severity, not p)"
        ),
    )


def compare_windows(
    baseline: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    model_name: str = "",
    model_version: str = "",
) -> DriftReport:
    """Compare two windows of inference records.

    Each record is a row from ``inference_logs``. Features are selected
    automatically by type: the predicted class and image format are categorical,
    dimensions and latency are continuous.
    """
    report = DriftReport(model_name=model_name, model_version=model_version)

    categorical = ("top_label", "image_format")
    continuous = ("image_width", "image_height", "image_bytes", "latency_ms")

    for feature in categorical:
        baseline_values = [r[feature] for r in baseline if r.get(feature) is not None]
        current_values = [r[feature] for r in current if r.get(feature) is not None]
        if baseline_values or current_values:
            report.results.append(
                detect_categorical_drift(feature, baseline_values, current_values)
            )

    for feature in continuous:
        baseline_values = [r[feature] for r in baseline if r.get(feature) is not None]
        current_values = [r[feature] for r in current if r.get(feature) is not None]
        if baseline_values or current_values:
            report.results.append(detect_continuous_drift(feature, baseline_values, current_values))

    logger.info(
        "Drift report for %s:%s -- severity=%s, drifted=%s",
        model_name,
        model_version,
        report.overall_severity.value,
        [r.feature for r in report.drifted_features] or "none",
    )
    return report
