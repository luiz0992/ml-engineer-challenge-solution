"""Tests for drift detection.

Statistics are verified against distributions whose correct answer is known, so
a broken implementation cannot pass by producing plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.validation.drift import (
    DriftSeverity,
    compare_windows,
    detect_categorical_drift,
    detect_continuous_drift,
    kolmogorov_smirnov,
    population_stability_index,
)

pytestmark = pytest.mark.unit


class TestPopulationStabilityIndex:
    def test_identical_distributions_score_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        baseline = rng.choice(["a", "b", "c"], 5000, p=[0.5, 0.3, 0.2]).tolist()
        current = rng.choice(["a", "b", "c"], 5000, p=[0.5, 0.3, 0.2]).tolist()

        psi, _ = population_stability_index(baseline, current)
        assert psi < 0.01

    def test_shifted_distribution_exceeds_the_significant_band(self) -> None:
        rng = np.random.default_rng(0)
        baseline = rng.choice(["a", "b", "c"], 5000, p=[0.5, 0.3, 0.2]).tolist()
        current = rng.choice(["a", "b", "c"], 5000, p=[0.2, 0.3, 0.5]).tolist()

        psi, _ = population_stability_index(baseline, current)
        assert psi > 0.25

    def test_a_new_category_produces_a_finite_score(self) -> None:
        """This is the case the epsilon guard exists for.

        A category present in one period and absent in the other gives ln(0) or
        a division by zero, making PSI infinite -- and it is precisely the case
        that matters most, since an appearing or vanishing class is strong
        evidence of drift.
        """
        baseline = ["a"] * 1000 + ["b"] * 1000
        current = ["a"] * 1000 + ["c"] * 1000

        psi, _ = population_stability_index(baseline, current)

        assert np.isfinite(psi)
        assert psi > 0.25

    def test_contributions_identify_the_moving_category(self) -> None:
        """A report must say *what* moved, not only that something did."""
        baseline = ["a"] * 800 + ["b"] * 200
        current = ["a"] * 200 + ["b"] * 800

        _, contributions = population_stability_index(baseline, current)

        assert set(contributions) == {"a", "b"}
        assert all(abs(v) > 0.1 for v in contributions.values())

    def test_is_symmetric_in_magnitude(self) -> None:
        baseline = ["a"] * 700 + ["b"] * 300
        current = ["a"] * 300 + ["b"] * 700

        forward, _ = population_stability_index(baseline, current)
        reverse, _ = population_stability_index(current, baseline)

        assert forward == pytest.approx(reverse, rel=1e-6)


class TestKolmogorovSmirnov:
    def test_same_distribution_gives_a_small_statistic(self) -> None:
        rng = np.random.default_rng(0)
        statistic, p_value = kolmogorov_smirnov(
            rng.normal(100, 15, 2000), rng.normal(100, 15, 2000)
        )
        assert statistic < 0.1
        assert p_value > 0.01

    def test_shifted_distribution_gives_a_large_statistic(self) -> None:
        rng = np.random.default_rng(0)
        statistic, p_value = kolmogorov_smirnov(
            rng.normal(100, 15, 2000), rng.normal(140, 15, 2000)
        )
        assert statistic > 0.5
        assert p_value < 0.001


class TestSeverityClassification:
    def test_categorical_severities(self) -> None:
        rng = np.random.default_rng(0)
        baseline = rng.choice(["a", "b", "c"], 3000, p=[0.5, 0.3, 0.2]).tolist()

        same = rng.choice(["a", "b", "c"], 3000, p=[0.5, 0.3, 0.2]).tolist()
        shifted = rng.choice(["a", "b", "c"], 3000, p=[0.1, 0.2, 0.7]).tolist()

        assert detect_categorical_drift("f", baseline, same).severity is DriftSeverity.NONE
        assert (
            detect_categorical_drift("f", baseline, shifted).severity is DriftSeverity.SIGNIFICANT
        )

    def test_continuous_severities(self) -> None:
        rng = np.random.default_rng(0)
        baseline = rng.normal(100, 15, 2000)

        assert (
            detect_continuous_drift("f", baseline, rng.normal(100, 15, 2000)).severity
            is DriftSeverity.NONE
        )
        assert (
            detect_continuous_drift("f", baseline, rng.normal(150, 15, 2000)).severity
            is DriftSeverity.SIGNIFICANT
        )

    def test_small_samples_report_insufficient_data(self) -> None:
        """A noisy estimate must not be presented as a finding."""
        result = detect_categorical_drift("f", ["a"] * 10, ["b"] * 10)

        assert result.severity is DriftSeverity.INSUFFICIENT_DATA
        assert not result.drifted

    def test_severity_is_driven_by_the_statistic_not_the_p_value(self) -> None:
        """At production volume a p-value rejects the null for trivial shifts.

        A tiny but consistent difference across 50,000 samples is overwhelmingly
        "significant" and entirely unimportant. Severity must come from the
        effect size.
        """
        rng = np.random.default_rng(0)
        baseline = rng.normal(100, 15, 50_000)
        barely_different = rng.normal(100.3, 15, 50_000)

        result = detect_continuous_drift("f", baseline, barely_different)

        assert result.p_value is not None
        assert result.severity is DriftSeverity.NONE, (
            f"a trivial shift was reported as {result.severity} "
            f"(statistic={result.statistic:.4f}, p={result.p_value:.2e})"
        )


class TestWindowComparison:
    @staticmethod
    def _records(count: int, *, label: str, width: int, latency: float) -> list[dict]:
        rng = np.random.default_rng(abs(hash(label)) % 2**32)
        return [
            {
                "top_label": label,
                "image_format": "JPEG",
                "image_width": int(rng.normal(width, 10)),
                "image_height": int(rng.normal(width, 10)),
                "image_bytes": 50_000,
                "latency_ms": float(rng.normal(latency, 1)),
            }
            for _ in range(count)
        ]

    def test_stable_traffic_reports_no_drift(self) -> None:
        baseline = self._records(500, label="cat", width=640, latency=10)
        current = self._records(500, label="cat", width=640, latency=10)

        report = compare_windows(baseline, current, model_name="m", model_version="v1")

        assert report.overall_severity is DriftSeverity.NONE
        assert not report.drifted_features

    def test_changed_predictions_are_detected(self) -> None:
        baseline = self._records(500, label="cat", width=640, latency=10)
        current = self._records(500, label="dog", width=640, latency=10)

        report = compare_windows(baseline, current)

        assert report.overall_severity is DriftSeverity.SIGNIFICANT
        assert "top_label" in [r.feature for r in report.drifted_features]

    def test_changed_input_dimensions_are_detected(self) -> None:
        """A client switching camera or resolution shows up here first."""
        baseline = self._records(500, label="cat", width=640, latency=10)
        current = self._records(500, label="cat", width=1920, latency=10)

        report = compare_windows(baseline, current)

        drifted = [r.feature for r in report.drifted_features]
        assert "image_width" in drifted

    def test_one_drifted_feature_makes_the_report_significant(self) -> None:
        """Averaging severity would dilute exactly the signal worth acting on."""
        baseline = self._records(500, label="cat", width=640, latency=10)
        current = self._records(500, label="cat", width=640, latency=10)
        current = [{**r, "top_label": "dog"} for r in current]

        report = compare_windows(baseline, current)

        assert report.overall_severity is DriftSeverity.SIGNIFICANT

    def test_report_serialises(self) -> None:
        report = compare_windows(
            self._records(200, label="cat", width=640, latency=10),
            self._records(200, label="cat", width=640, latency=10),
            model_name="m",
            model_version="v1",
        )
        payload = report.as_dict()

        assert payload["model_name"] == "m"
        assert "results" in payload
        assert all("severity" in r for r in payload["results"])
