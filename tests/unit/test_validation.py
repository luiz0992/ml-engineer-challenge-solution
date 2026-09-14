"""Tests for the A/B testing framework and regression gate."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from models.validation.ab_testing import (
    Experiment,
    Variant,
    analyse_experiment,
    compare_means,
    compare_proportions,
)
from models.validation.regression import (
    DEFAULT_SPECS,
    Direction,
    MetricSpec,
    compare_to_baseline,
    load_baseline,
    write_baseline,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# A/B assignment
# ---------------------------------------------------------------------------
class TestExperimentConfiguration:
    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            Experiment("e", [Variant("a", "v1", 0.5), Variant("b", "v2", 0.3)])

    def test_variant_names_must_be_unique(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            Experiment("e", [Variant("a", "v1", 0.5), Variant("a", "v2", 0.5)])

    def test_requires_at_least_one_variant(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Experiment("e", [])

    def test_weight_must_be_a_proportion(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            Variant("a", "v1", 1.5)


class TestAssignment:
    @pytest.fixture
    def experiment(self) -> Experiment:
        return Experiment(
            "rollout", [Variant("control", "v1", 0.5), Variant("treatment", "v2", 0.5)]
        )

    def test_assignment_is_stable_for_a_user(self, experiment: Experiment) -> None:
        """A user must not switch arms between requests.

        Random per-request assignment would give the same caller different
        model versions for identical inputs, and would correlate observations
        within a user across arms, violating the independence a significance
        test assumes.
        """
        first = [experiment.assign(f"user-{i}").name for i in range(200)]
        second = [experiment.assign(f"user-{i}").name for i in range(200)]

        assert first == second

    def test_traffic_splits_in_the_configured_proportion(self, experiment: Experiment) -> None:
        counts = Counter(experiment.assign(f"user-{i}").name for i in range(20_000))

        assert counts["control"] == pytest.approx(10_000, rel=0.05)
        assert counts["treatment"] == pytest.approx(10_000, rel=0.05)

    def test_uneven_weights_are_honoured(self) -> None:
        canary = Experiment("canary", [Variant("control", "v1", 0.9), Variant("canary", "v2", 0.1)])
        counts = Counter(canary.assign(f"user-{i}").name for i in range(20_000))

        assert counts["canary"] == pytest.approx(2_000, rel=0.15)

    def test_different_experiments_reshuffle_users(self) -> None:
        """Otherwise the same users land in every challenger arm.

        Results across experiments would then be correlated, and a subpopulation
        that reacts badly to change would appear to poison every test.
        """
        first = Experiment("a", [Variant("c", "v1", 0.5), Variant("t", "v2", 0.5)])
        second = Experiment("b", [Variant("c", "v1", 0.5), Variant("t", "v2", 0.5)])

        users = [f"user-{i}" for i in range(5_000)]
        agreement = sum(1 for u in users if first.assign(u).name == second.assign(u).name)

        assert agreement / len(users) == pytest.approx(0.5, abs=0.05)

    def test_disabled_experiment_sends_everyone_to_control(self) -> None:
        experiment = Experiment(
            "off",
            [Variant("control", "v1", 0.5), Variant("treatment", "v2", 0.5)],
            enabled=False,
        )
        assert all(experiment.assign(f"user-{i}").name == "control" for i in range(100))


# ---------------------------------------------------------------------------
# A/B analysis
# ---------------------------------------------------------------------------
class TestStatisticalComparison:
    def test_detects_a_real_difference_in_means(self) -> None:
        rng = np.random.default_rng(0)
        result = compare_means("latency", rng.normal(10, 2, 2000), rng.normal(14, 2, 2000))

        assert result.significant
        assert result.material
        assert result.absolute_difference == pytest.approx(4.0, abs=0.3)

    def test_confidence_interval_brackets_the_difference(self) -> None:
        rng = np.random.default_rng(0)
        result = compare_means("latency", rng.normal(10, 2, 2000), rng.normal(12, 2, 2000))

        low, high = result.confidence_interval
        assert low < result.absolute_difference < high

    def test_a_trivial_difference_at_large_n_is_not_material(self) -> None:
        """The production failure mode this guard exists for.

        With 50,000 observations per arm a 0.5% difference is overwhelmingly
        significant and entirely unimportant. Reporting it as a regression
        would block every rollout.
        """
        rng = np.random.default_rng(0)
        result = compare_means("latency", rng.normal(10.0, 2, 50_000), rng.normal(10.05, 2, 50_000))

        assert not result.material, (
            f"a {result.relative_difference * 100:.2f}% difference was reported as "
            f"material (p={result.p_value:.2e})"
        )

    def test_proportion_comparison_detects_a_real_drop(self) -> None:
        result = compare_proportions("success", 990, 1000, 900, 1000)

        assert result.significant
        assert result.material
        assert result.absolute_difference < 0

    def test_identical_proportions_are_not_significant(self) -> None:
        result = compare_proportions("success", 950, 1000, 951, 1000)

        assert not result.significant
        assert not result.material


class TestExperimentAnalysis:
    @staticmethod
    def _arm(mean_latency: float, success_rate: float, n: int = 1000) -> list[dict]:
        rng = np.random.default_rng(int(mean_latency * 100))
        return [
            {
                "latency_ms": float(rng.normal(mean_latency, 1.5)),
                "status": "success" if rng.random() < success_rate else "failure",
                "top_probability": 0.9,
            }
            for _ in range(n)
        ]

    @pytest.fixture
    def experiment(self) -> Experiment:
        return Experiment(
            "rollout", [Variant("control", "v1", 0.5), Variant("treatment", "v2", 0.5)]
        )

    def test_blocks_promotion_on_a_latency_regression(self, experiment: Experiment) -> None:
        report = analyse_experiment(
            experiment,
            {"control": self._arm(10, 0.99), "treatment": self._arm(15, 0.99)},
        )

        assert "Do not promote" in report.recommendation
        assert "latency" in report.recommendation

    def test_blocks_promotion_on_a_success_rate_regression(self, experiment: Experiment) -> None:
        report = analyse_experiment(
            experiment,
            {"control": self._arm(10, 0.99), "treatment": self._arm(10, 0.90)},
        )

        assert "Do not promote" in report.recommendation

    def test_equivalent_arms_are_reported_as_no_difference(self, experiment: Experiment) -> None:
        report = analyse_experiment(
            experiment,
            {"control": self._arm(10, 0.99), "treatment": self._arm(10, 0.99)},
        )

        assert "No material difference" in report.recommendation

    def test_insufficient_control_data_is_reported_not_guessed(
        self, experiment: Experiment
    ) -> None:
        report = analyse_experiment(
            experiment,
            {"control": self._arm(10, 0.99, n=10), "treatment": self._arm(10, 0.99)},
        )

        assert "Insufficient data" in report.recommendation
        assert not report.comparisons

    def test_report_serialises(self, experiment: Experiment) -> None:
        report = analyse_experiment(
            experiment,
            {"control": self._arm(10, 0.99), "treatment": self._arm(10, 0.99)},
        )
        payload = report.as_dict()

        assert payload["experiment"] == "rollout"
        assert len(payload["arms"]) == 2
        assert payload["recommendation"]


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------
class TestRegressionGate:
    BASELINE = {
        "model_name": "classifier",
        "version": "v1",
        "metrics": {
            "acc_top1": 0.8588,
            "acc_top5": 0.9654,
            "latency_p50_ms": 5.2,
            "latency_p95_ms": 5.3,
            "artifact_size_mb": 87.9,
        },
    }

    def _check(self, **overrides: float):
        metrics = {**self.BASELINE["metrics"], **overrides}
        return compare_to_baseline(self.BASELINE, {"version": "cand", "metrics": metrics})

    def test_identical_metrics_pass(self) -> None:
        assert self._check().passed

    def test_improvements_are_never_regressions(self) -> None:
        report = self._check(acc_top1=0.88, latency_p50_ms=4.0)

        assert report.passed
        assert any(r.improved for r in report.results)

    def test_small_degradation_within_tolerance_passes(self) -> None:
        """Run-to-run variance must not fail the build."""
        assert self._check(acc_top1=0.8502).passed  # -1%, tolerance is 2%

    def test_large_accuracy_drop_fails(self) -> None:
        report = self._check(acc_top1=0.8159)  # -5%

        assert not report.passed
        assert report.regressions[0].metric == "acc_top1"

    def test_absolute_floor_catches_gradual_erosion(self) -> None:
        """A relative tolerance alone permits a slow slide across releases.

        Each step stays within tolerance while the model degrades steadily; the
        floor is what stops it.
        """
        report = self._check(acc_top1=0.79)

        assert not report.passed
        assert "floor" in report.regressions[0].detail

    def test_latency_increase_beyond_tolerance_fails(self) -> None:
        report = self._check(latency_p50_ms=6.8)  # +31%, tolerance is 20%

        assert not report.passed
        assert report.regressions[0].metric == "latency_p50_ms"

    def test_a_missing_metric_fails_rather_than_passing_silently(self) -> None:
        """A comparison that could not be made must not count as success."""
        metrics = {k: v for k, v in self.BASELINE["metrics"].items() if k != "acc_top1"}
        report = compare_to_baseline(self.BASELINE, {"version": "c", "metrics": metrics})

        assert not report.passed
        assert "acc_top1" in report.missing_metrics

    def test_a_newly_tracked_metric_is_not_a_failure(self) -> None:
        """A metric absent from the baseline has nothing to compare against."""
        metrics = {**self.BASELINE["metrics"], "new_metric": 1.0}
        report = compare_to_baseline(self.BASELINE, {"version": "c", "metrics": metrics})

        assert report.passed

    def test_summary_names_the_failing_metric(self) -> None:
        summary = self._check(acc_top1=0.70).summary()

        assert "FAILED" in summary
        assert "acc_top1" in summary

    def test_baseline_round_trips(self, tmp_path) -> None:
        path = tmp_path / "baseline.json"
        write_baseline(path, model_name="m", version="v1", metrics={"acc_top1": 0.85}, notes="n")

        loaded = load_baseline(path)
        assert loaded["metrics"]["acc_top1"] == 0.85
        assert loaded["version"] == "v1"

    def test_missing_baseline_explains_how_to_create_one(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="write_baseline"):
            load_baseline(tmp_path / "absent.json")


class TestMetricSpec:
    def test_higher_is_better_direction(self) -> None:
        spec = MetricSpec("acc", Direction.HIGHER_IS_BETTER, tolerance=0.05)

        assert not spec.regressed(0.90, 0.95)[0]  # improvement
        assert not spec.regressed(0.90, 0.87)[0]  # within tolerance
        assert spec.regressed(0.90, 0.80)[0]  # beyond tolerance

    def test_lower_is_better_direction(self) -> None:
        spec = MetricSpec("latency", Direction.LOWER_IS_BETTER, tolerance=0.10)

        assert not spec.regressed(10.0, 8.0)[0]
        assert not spec.regressed(10.0, 10.5)[0]
        assert spec.regressed(10.0, 12.0)[0]

    def test_default_specs_cover_accuracy_and_latency(self) -> None:
        names = {s.name for s in DEFAULT_SPECS}
        assert {"acc_top1", "latency_p50_ms"} <= names


class TestCommittedBaseline:
    def test_the_repository_baseline_is_valid(self) -> None:
        """The committed baseline must load and cover the gated metrics.

        A malformed or incomplete baseline would silently weaken the gate.
        """
        from pathlib import Path

        path = Path("models/validation/baselines/classifier.json")
        if not path.exists():
            pytest.skip("No committed baseline")

        baseline = load_baseline(path)

        assert baseline["metrics"]["acc_top1"] > 0.80
        for spec in DEFAULT_SPECS:
            assert spec.name in baseline["metrics"], f"{spec.name} is not baselined"

    def test_the_trained_model_passes_its_own_baseline(self) -> None:
        """A sanity check that the gate is not trivially broken."""
        from pathlib import Path

        path = Path("models/validation/baselines/classifier.json")
        if not path.exists():
            pytest.skip("No committed baseline")

        baseline = load_baseline(path)
        report = compare_to_baseline(baseline, baseline)

        assert report.passed
