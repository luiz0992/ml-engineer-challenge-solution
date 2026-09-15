"""Tests for A/B experiment routing in the serving path.

The behaviour that matters is what happens when an experiment is
misconfigured. An experiment is an optimisation; failing requests over one
would be absurd, so every bad-configuration path is asserted to degrade to the
active model version rather than error.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from api.services.experiment_service import ExperimentRegistry

pytestmark = pytest.mark.unit


def _config(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
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
    base["experiments"][0].update(overrides)
    return base


AVAILABLE = {"classifier": {"v1", "v2"}}


class _CapturingFactory:
    """Stands in for the session factory, keeping the rows that were written."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(self) -> Any:
        return self

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def add_all(self, objects: list[Any]) -> None:
        self.rows.extend({"variant": o.variant, "model_version": o.model_version} for o in objects)

    async def commit(self) -> None:
        return None


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


class TestVariantReachesTheAuditTrail:
    """The end-to-end property the A/B framework exists for.

    Routing traffic is only half of it. An experiment is analysed by grouping
    `inference_logs` on `variant`, so a run that splits traffic correctly but
    records the wrong variant -- or none -- produces a comparison between two
    arms that are not the arms that served the traffic. That failure is silent
    and would invalidate the result rather than break the request.

    So this drives real requests through the app with two model versions
    registered and an experiment across them, then asserts that every audit
    record's `(variant, model_version)` pair matches the version that served it.
    """

    USERS = 60

    @pytest.fixture
    async def ab_app(self, app: Any) -> Any:
        """An app serving two versions under one experiment, with audit capture.

        `v2` is registered as a second entry pointing at the same session. The
        artifact pipeline only produces one version (see the README's known
        limitations), and the property under test is the registry-to-audit
        path, not that the two versions differ numerically.

        One API key per simulated user, because assignment hashes `user_id`:
        driving 60 requests from a single principal would put every one of them
        in the same arm and make the split untestable. It also gives each user
        their own rate-limit bucket.
        """
        from api.config import UserTier
        from api.main import lifespan
        from api.services.experiment_service import ExperimentRegistry

        async with lifespan(app):
            loaded = app.state.model_service.get_classifier()
            app.state.model_service.register(replace(loaded, version="v2"), make_active=False)

            path = Path(app.state.settings.artifacts_dir) / "experiments.json"
            path.write_text(json.dumps(_config(model_name=loaded.name)))
            app.state.experiment_registry = ExperimentRegistry.load(
                path, available_versions={loaded.name: {"v1", "v2"}}
            )

            for index in range(self.USERS):
                app.state.api_key_store.add(
                    f"key-user-{index}",
                    user_id=f"user-{index}",
                    tier=UserTier.ENTERPRISE,
                )

            app.state.audit_service._session_factory = _CapturingFactory()
            yield app

    @pytest.fixture
    async def ab_client(self, ab_app: Any) -> Any:
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=ab_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    async def _drive(self, client: Any) -> None:
        """One classification per simulated user, bypassing the cache.

        A cached response short-circuits inference, and the question here is
        what the serving path records.
        """
        from tests.conftest import make_image_bytes

        image = make_image_bytes()
        for index in range(self.USERS):
            token = await client.post("/api/v1/auth/token", json={"api_key": f"key-user-{index}"})
            assert token.status_code == 200

            response = await client.post(
                "/api/v1/classify",
                files={"file": (f"{index}.jpg", image, "image/jpeg")},
                headers={"Authorization": f"Bearer {token.json()['access_token']}"},
                params={"use_cache": "false"},
            )
            assert response.status_code == 200, response.text

    async def test_every_audit_row_carries_the_serving_variant(
        self, ab_app: Any, ab_client: Any
    ) -> None:
        await self._drive(ab_client)
        await ab_app.state.audit_service.flush()
        rows = ab_app.state.audit_service._session_factory.rows

        assert len(rows) == self.USERS, "every request must produce exactly one audit row"

        for row in rows:
            assert row["variant"] in {"control", "treatment"}
            expected = "v1" if row["variant"] == "control" else "v2"
            assert row["model_version"] == expected, (
                f"variant {row['variant']} was recorded against version "
                f"{row['model_version']}; the A/B comparison would attribute "
                f"these results to the wrong arm"
            )

    async def test_traffic_actually_splits(self, ab_app: Any, ab_client: Any) -> None:
        """A split that never fires would make the test above vacuous."""
        await self._drive(ab_client)
        await ab_app.state.audit_service.flush()
        rows = ab_app.state.audit_service._session_factory.rows

        versions = Counter(row["model_version"] for row in rows)
        assert set(versions) == {"v1", "v2"}, f"only one arm was exercised: {versions}"

        # A 50/50 split over 60 users; anything beyond 20/40 would suggest the
        # hash is not distributing rather than ordinary sampling noise.
        assert min(versions.values()) >= 20, f"split is too lopsided: {versions}"
