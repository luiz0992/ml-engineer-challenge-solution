"""Tests for A/B experiment routing in the serving path.

The behaviour that matters is what happens when an experiment is
misconfigured. An experiment is an optimisation; failing requests over one
would be absurd, so every bad-configuration path is asserted to degrade to the
active model version rather than error.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from api.services.experiment_service import ExperimentRegistry

pytestmark = pytest.mark.unit


def _config(**overrides: object) -> dict:
    base = {
        "experiments": [
            {
                "name": "rollout",
                "model_name": "classifier",
                "enabled": True,
                "variants": [
                    {"name": "control", "model_version": "v1", "weight": 0.5},
                    {"name": "treatment", "model_version": "v2", "weight": 0.5},
                ],
            }
        ]
    }
    base["experiments"][0].update(overrides)  # type: ignore[index]
    return base


AVAILABLE = {"classifier": {"v1", "v2"}}


class TestLoading:
    def test_loads_a_valid_experiment(self, tmp_path: Path) -> None:
        path = tmp_path / "experiments.json"
        path.write_text(json.dumps(_config()))

        registry = ExperimentRegistry.load(path, available_versions=AVAILABLE)

        assert registry.active_count == 1

    def test_absent_file_is_the_normal_case(self, tmp_path: Path) -> None:
        registry = ExperimentRegistry.load(tmp_path / "nothing.json", available_versions=AVAILABLE)
        assert registry.active_count == 0

    def test_malformed_json_does_not_raise(self, tmp_path: Path) -> None:
        """A broken config must not take down serving."""
        path = tmp_path / "experiments.json"
        path.write_text("{ this is not json")

        registry = ExperimentRegistry.load(path, available_versions=AVAILABLE)

        assert registry.active_count == 0

    def test_weights_that_do_not_sum_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "experiments.json"
        path.write_text(
            json.dumps(
                _config(
                    variants=[
                        {"name": "a", "model_version": "v1", "weight": 0.3},
                        {"name": "b", "model_version": "v2", "weight": 0.3},
                    ]
                )
            )
        )

        registry = ExperimentRegistry.load(path, available_versions=AVAILABLE)
        assert registry.active_count == 0

    def test_missing_fields_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "experiments.json"
        path.write_text(json.dumps({"experiments": [{"name": "broken"}]}))

        registry = ExperimentRegistry.load(path, available_versions=AVAILABLE)
        assert registry.active_count == 0

    def test_variant_naming_an_unloaded_version_is_rejected(self, tmp_path: Path) -> None:
        """Otherwise that share of traffic 404s.

        A variant pointing at a version that was never loaded would send half
        the users to a model that does not exist, which is worse than running
        no experiment at all.
        """
        path = tmp_path / "experiments.json"
        path.write_text(json.dumps(_config()))

        registry = ExperimentRegistry.load(path, available_versions={"classifier": {"v1"}})

        assert registry.active_count == 0

    def test_one_bad_experiment_does_not_disable_the_others(self, tmp_path: Path) -> None:
        path = tmp_path / "experiments.json"
        path.write_text(
            json.dumps(
                {
                    "experiments": [
                        {"name": "broken"},
                        _config()["experiments"][0],
                    ]
                }
            )
        )

        registry = ExperimentRegistry.load(path, available_versions=AVAILABLE)
        assert registry.active_count == 1


class TestResolution:
    @pytest.fixture
    def registry(self, tmp_path: Path) -> ExperimentRegistry:
        path = tmp_path / "experiments.json"
        path.write_text(json.dumps(_config()))
        return ExperimentRegistry.load(path, available_versions=AVAILABLE)

    def test_no_experiment_returns_no_override(self, registry: ExperimentRegistry) -> None:
        """The overwhelmingly common path: the active version serves."""
        assert registry.resolve_version("other-model", "alice") == (None, None)

    def test_resolves_to_a_variant(self, registry: ExperimentRegistry) -> None:
        version, variant = registry.resolve_version("classifier", "alice")

        assert version in {"v1", "v2"}
        assert variant in {"control", "treatment"}

    def test_resolution_is_stable_per_user(self, registry: ExperimentRegistry) -> None:
        first = registry.resolve_version("classifier", "alice")
        second = registry.resolve_version("classifier", "alice")

        assert first == second

    def test_traffic_splits_across_variants(self, registry: ExperimentRegistry) -> None:
        counts = Counter(
            registry.resolve_version("classifier", f"user-{i}")[1] for i in range(4000)
        )

        assert counts["control"] == pytest.approx(2000, rel=0.1)
        assert counts["treatment"] == pytest.approx(2000, rel=0.1)

    def test_version_and_variant_stay_paired(self, registry: ExperimentRegistry) -> None:
        """A mismatch would attribute results to the wrong arm."""
        pairs = {registry.resolve_version("classifier", f"user-{i}") for i in range(500)}

        assert pairs <= {("v1", "control"), ("v2", "treatment")}

    def test_disabled_experiment_sends_all_traffic_to_control(self, tmp_path: Path) -> None:
        path = tmp_path / "experiments.json"
        path.write_text(json.dumps(_config(enabled=False)))
        registry = ExperimentRegistry.load(path, available_versions=AVAILABLE)

        variants = {registry.resolve_version("classifier", f"user-{i}")[1] for i in range(200)}
        assert variants == {"control"}


class TestServingIntegration:
    async def test_explicit_pin_overrides_the_experiment(
        self, client, auth_headers: dict[str, str]
    ) -> None:
        """Version pinning must win, or it is not pinning.

        A caller asking for a specific version and silently receiving another
        makes the API's versioning contract a lie.
        """
        from tests.conftest import make_image_bytes

        response = await client.post(
            "/api/v1/classify?model_version=v1",
            files={"file": ("a.jpg", make_image_bytes(), "image/jpeg")},
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.json()["provenance"]["model_version"] == "v1"

    async def test_requests_succeed_with_no_experiments_configured(
        self, client, auth_headers: dict[str, str]
    ) -> None:
        from tests.conftest import make_image_bytes

        response = await client.post(
            "/api/v1/classify",
            files={"file": ("a.jpg", make_image_bytes(), "image/jpeg")},
            headers=auth_headers,
        )
        assert response.status_code == 200
