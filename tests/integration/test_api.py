"""Integration tests exercising the full request path.

These run against the assembled application — routers, middleware, services,
and a fake Redis — with only the model replaced by a synthetic graph. They
verify behaviour a unit test cannot: middleware ordering, dependency wiring,
error envelope consistency, and the interaction between auth, rate limiting,
and caching.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.conftest import make_image_bytes

pytestmark = pytest.mark.integration

CLASSIFY = "/api/v1/classify"


def upload(data: bytes, filename: str = "image.jpg", content_type: str = "image/jpeg"):
    return {"file": (filename, data, content_type)}


class TestHealthAndMetadata:
    async def test_health_reports_components(self, client: Any) -> None:
        response = await client.get("/api/v1/health")
        body = response.json()

        assert response.status_code == 200
        assert body["status"] == "healthy"
        assert {c["name"] for c in body["components"]} >= {"models", "cache"}

    async def test_liveness_needs_no_dependencies(self, client: Any) -> None:
        """Liveness must not consult the cache or database.

        Tying it to dependencies restarts healthy processes when something
        downstream fails, turning a partial outage into a crash loop.
        """
        assert (await client.get("/health/live")).status_code == 200

    async def test_readiness_reflects_model_state(self, client: Any) -> None:
        body = (await client.get("/health/ready")).json()
        assert body["ready"] is True
        assert body["models_loaded"] >= 1

    async def test_models_endpoint_exposes_provenance(self, client: Any) -> None:
        body = (await client.get("/api/v1/models")).json()
        model = body["models"][0]

        assert model["loaded"] is True
        assert model["is_active"] is True
        assert model["num_classes"] == 10
        assert model["metrics"]["final_acc_top1"] == 0.85

    async def test_metrics_are_prometheus_formatted(self, client: Any) -> None:
        response = await client.get("/api/v1/metrics")
        assert response.status_code == 200
        assert "http_requests_total" in response.text
        assert "model_loaded_info" in response.text


class TestAuthentication:
    async def test_inference_requires_a_token(self, client: Any) -> None:
        response = await client.post(CLASSIFY, files=upload(make_image_bytes()))

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "authentication_required"

    async def test_rejects_a_malformed_token(self, client: Any) -> None:
        response = await client.post(
            CLASSIFY,
            files=upload(make_image_bytes()),
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert response.status_code == 401

    async def test_token_exchange_returns_the_caller_tier(self, client: Any) -> None:
        body = (
            await client.post("/api/v1/auth/token", json={"api_key": "dev-key-enterprise"})
        ).json()

        assert body["tier"] == "enterprise"
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0

    async def test_unknown_api_key_is_rejected(self, client: Any) -> None:
        response = await client.post("/api/v1/auth/token", json={"api_key": "not-a-key"})
        assert response.status_code == 401

    async def test_rejection_does_not_reveal_whether_a_key_exists(self, client: Any) -> None:
        """Both failures must look identical to a caller.

        A distinguishable response lets an attacker enumerate valid keys.
        """
        unknown = await client.post("/api/v1/auth/token", json={"api_key": "aaaaaaaaaa"})
        malformed = await client.post("/api/v1/auth/token", json={"api_key": "bbbbbbbbbb"})

        assert unknown.status_code == malformed.status_code
        assert unknown.json()["error"]["message"] == malformed.json()["error"]["message"]


class TestClassification:
    async def test_returns_ranked_predictions_with_provenance(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            f"{CLASSIFY}?top_k=3", files=upload(make_image_bytes()), headers=auth_headers
        )
        body = response.json()

        assert response.status_code == 200
        assert len(body["predictions"]) == 3

        probabilities = [p["probability"] for p in body["predictions"]]
        assert probabilities == sorted(probabilities, reverse=True)

        # Provenance makes a recorded prediction attributable to an artefact.
        assert body["provenance"]["model_name"] == "tiny-imagenet-classifier"
        assert body["provenance"]["backend"]
        assert body["correlation_id"]

    async def test_predictions_carry_labels_not_just_indices(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        body = (
            await client.post(CLASSIFY, files=upload(make_image_bytes()), headers=auth_headers)
        ).json()
        prediction = body["predictions"][0]

        assert prediction["label"].startswith("class_")
        assert isinstance(prediction["class_id"], int)
        assert prediction["wnid"]

    async def test_probabilities_can_be_omitted(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        body = (
            await client.post(
                f"{CLASSIFY}?include_probabilities=false",
                files=upload(make_image_bytes()),
                headers=auth_headers,
            )
        ).json()
        assert body["predictions"][0]["probability"] is None

    @pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "BMP"])
    async def test_accepts_every_supported_format(
        self, client: Any, auth_headers: dict[str, str], fmt: str
    ) -> None:
        response = await client.post(
            CLASSIFY, files=upload(make_image_bytes(fmt=fmt)), headers=auth_headers
        )
        assert response.status_code == 200

    async def test_rate_limit_headers_are_present_on_success(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            CLASSIFY, files=upload(make_image_bytes()), headers=auth_headers
        )
        assert int(response.headers["X-RateLimit-Limit"]) > 0
        assert "X-RateLimit-Remaining" in response.headers


class TestValidationErrors:
    @pytest.mark.parametrize(
        ("payload", "expected_code"),
        [
            (b"not an image", "invalid_image"),
            (b"", "invalid_image"),
        ],
    )
    async def test_rejects_unusable_uploads(
        self, client: Any, auth_headers: dict[str, str], payload: bytes, expected_code: str
    ) -> None:
        response = await client.post(CLASSIFY, files=upload(payload), headers=auth_headers)

        assert response.status_code == 422
        assert response.json()["error"]["code"] == expected_code

    async def test_rejects_unsupported_format_with_guidance(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            CLASSIFY,
            files=upload(make_image_bytes(fmt="TIFF"), "x.tiff", "image/tiff"),
            headers=auth_headers,
        )
        body = response.json()

        assert response.status_code == 422
        assert body["error"]["code"] == "unsupported_format"
        # The caller is told what is accepted, not merely what was refused.
        assert "JPEG" in body["error"]["details"]["supported_formats"]

    async def test_content_type_is_not_trusted(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        """A TIFF declared as JPEG is still rejected.

        Format is determined from the file's own bytes; trusting the header
        would let a caller reach an unexpected decoder.
        """
        response = await client.post(
            CLASSIFY,
            files=upload(make_image_bytes(fmt="TIFF"), "x.jpg", "image/jpeg"),
            headers=auth_headers,
        )
        assert response.json()["error"]["code"] == "unsupported_format"

    async def test_rejects_oversized_upload(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            CLASSIFY, files=upload(b"\xff" * (6 * 1024 * 1024)), headers=auth_headers
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    @pytest.mark.parametrize("top_k", [0, -1, 1000])
    async def test_validates_query_parameters(
        self, client: Any, auth_headers: dict[str, str], top_k: int
    ) -> None:
        response = await client.post(
            f"{CLASSIFY}?top_k={top_k}", files=upload(make_image_bytes()), headers=auth_headers
        )
        assert response.status_code == 422

    async def test_unknown_model_version_returns_404(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            f"{CLASSIFY}?model_version=v99",
            files=upload(make_image_bytes()),
            headers=auth_headers,
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"


class TestErrorEnvelope:
    async def test_every_error_uses_the_same_shape(self, client: Any) -> None:
        """One envelope for all errors, including framework-generated ones.

        FastAPI's default validation body differs from the application's, which
        would force clients to parse two formats.
        """
        responses = [
            await client.post(CLASSIFY, files=upload(make_image_bytes())),  # 401
            await client.get("/api/v1/no-such-route"),  # 404
            await client.post("/api/v1/auth/token", json={}),  # 422
        ]

        for response in responses:
            error = response.json()["error"]
            assert isinstance(error["code"], str)
            assert isinstance(error["message"], str)

    async def test_errors_carry_a_correlation_id(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(CLASSIFY, files=upload(b"bad"), headers=auth_headers)
        assert response.json()["error"]["correlation_id"]

    async def test_internal_detail_is_not_leaked(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        """Error messages must not expose paths, modules, or stack frames."""
        body = (
            await client.post(CLASSIFY, files=upload(b"\x00" * 100), headers=auth_headers)
        ).json()
        message = body["error"]["message"].lower()

        for leak in ("traceback", "/app/", "site-packages", 'file "'):
            assert leak not in message


class TestCorrelationId:
    async def test_generated_when_absent(self, client: Any) -> None:
        response = await client.get("/api/v1/health")
        assert len(response.headers["X-Correlation-ID"]) >= 16

    async def test_inbound_id_is_adopted(self, client: Any) -> None:
        """A trace started upstream must survive into this service."""
        response = await client.get(
            "/api/v1/health", headers={"X-Correlation-ID": "trace-from-caller"}
        )
        assert response.headers["X-Correlation-ID"] == "trace-from-caller"

    async def test_oversized_inbound_id_is_truncated(self, client: Any) -> None:
        """An unbounded client-supplied value is a log-injection vector."""
        response = await client.get("/api/v1/health", headers={"X-Correlation-ID": "x" * 5000})
        assert len(response.headers["X-Correlation-ID"]) <= 64

    async def test_ids_are_unique_across_concurrent_requests(self, client: Any) -> None:
        """Concurrent requests must not share a correlation ID.

        The ID lives in a ContextVar; if it were module-level state, concurrent
        async handlers would overwrite each other and logs would be
        unattributable.
        """
        responses = await asyncio.gather(*(client.get("/api/v1/health") for _ in range(10)))
        ids = {r.headers["X-Correlation-ID"] for r in responses}
        assert len(ids) == 10


class TestCaching:
    async def test_repeat_request_is_served_from_cache(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        image = make_image_bytes(seed=42)

        first = await client.post(CLASSIFY, files=upload(image), headers=auth_headers)
        second = await client.post(CLASSIFY, files=upload(image), headers=auth_headers)

        assert first.json()["cached"] is False
        assert second.json()["cached"] is True
        assert first.json()["predictions"] == second.json()["predictions"]

    async def test_different_images_do_not_collide(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        await client.post(CLASSIFY, files=upload(make_image_bytes(seed=1)), headers=auth_headers)
        second = await client.post(
            CLASSIFY, files=upload(make_image_bytes(seed=2)), headers=auth_headers
        )
        assert second.json()["cached"] is False

    async def test_options_are_part_of_the_cache_key(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        """A top_k=1 result must not be served to a top_k=5 request."""
        image = make_image_bytes(seed=7)

        await client.post(f"{CLASSIFY}?top_k=1", files=upload(image), headers=auth_headers)
        second = await client.post(f"{CLASSIFY}?top_k=5", files=upload(image), headers=auth_headers)

        assert second.json()["cached"] is False
        assert len(second.json()["predictions"]) == 5

    async def test_cache_can_be_bypassed(self, client: Any, auth_headers: dict[str, str]) -> None:
        image = make_image_bytes(seed=9)

        await client.post(CLASSIFY, files=upload(image), headers=auth_headers)
        second = await client.post(
            f"{CLASSIFY}?use_cache=false", files=upload(image), headers=auth_headers
        )
        assert second.json()["cached"] is False


class TestRateLimiting:
    async def test_quota_is_enforced_per_tier(self, client: Any, settings: Any) -> None:
        token = (await client.post("/api/v1/auth/token", json={"api_key": "dev-key-free"})).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}

        limit = settings.rate_limit_free_rpm
        statuses = []
        for index in range(limit + 3):
            # Vary top_k so the cache does not serve a repeat and bypass
            # nothing — the limiter runs before the cache either way, but this
            # keeps the test honest about what it exercises.
            response = await client.post(
                f"{CLASSIFY}?top_k={(index % 9) + 1}",
                files=upload(make_image_bytes(seed=index)),
                headers=headers,
            )
            statuses.append(response.status_code)

        assert statuses.count(200) == limit
        assert statuses.count(429) == 3

    async def test_rejection_includes_retry_after(self, client: Any, settings: Any) -> None:
        token = (await client.post("/api/v1/auth/token", json={"api_key": "dev-key-free"})).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}

        response = None
        for index in range(settings.rate_limit_free_rpm + 2):
            response = await client.post(
                CLASSIFY, files=upload(make_image_bytes(seed=index)), headers=headers
            )

        assert response is not None
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) > 0

    async def test_tiers_are_isolated(self, client: Any, settings: Any) -> None:
        """One tier exhausting its quota must not affect another."""
        free_token = (
            await client.post("/api/v1/auth/token", json={"api_key": "dev-key-free"})
        ).json()["access_token"]

        for index in range(settings.rate_limit_free_rpm + 2):
            await client.post(
                CLASSIFY,
                files=upload(make_image_bytes(seed=index)),
                headers={"Authorization": f"Bearer {free_token}"},
            )

        pro_token = (
            await client.post("/api/v1/auth/token", json={"api_key": "dev-key-pro"})
        ).json()["access_token"]
        response = await client.post(
            CLASSIFY,
            files=upload(make_image_bytes()),
            headers={"Authorization": f"Bearer {pro_token}"},
        )
        assert response.status_code == 200


class TestDetection:
    async def test_reports_unavailable_rather_than_missing(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        """503, not 404.

        A caller must be able to distinguish "not deployed here" from "you
        called the wrong URL".
        """
        response = await client.post(
            "/api/v1/detect", files=upload(make_image_bytes()), headers=auth_headers
        )

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        # The message points at what does work.
        assert "classify" in response.json()["error"]["message"]


class TestOpenAPI:
    async def test_schema_documents_every_endpoint(self, client: Any) -> None:
        paths = (await client.get("/openapi.json")).json()["paths"]

        for required in (
            "/api/v1/classify",
            "/api/v1/detect",
            "/api/v1/batch",
            "/api/v1/models",
            "/api/v1/health",
            "/api/v1/metrics",
        ):
            assert required in paths, f"{required} is missing from the OpenAPI schema"

    async def test_error_responses_are_documented(self, client: Any) -> None:
        """Documented failure modes, so clients can handle them deliberately."""
        responses = (await client.get("/openapi.json")).json()["paths"]["/api/v1/classify"]["post"][
            "responses"
        ]

        for status in ("401", "413", "422", "429", "503"):
            assert status in responses


class TestMetricLabels:
    """Metric cardinality and correctness.

    Metrics that exist but are mislabelled are worse than none: they look
    healthy and make every dashboard wrong.
    """

    async def test_requests_are_labelled_by_route_template(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        """Labels must be the route template, not 'unmatched'.

        Starlette populates ``scope["route"]`` during routing, so reading it
        before ``call_next`` returns "unmatched" for every request and
        collapses the whole metric into one series.
        """
        await client.get("/api/v1/health")
        await client.get("/api/v1/models")

        body = (await client.get("/api/v1/metrics")).text
        counters = [line for line in body.splitlines() if line.startswith("http_requests_total{")]

        assert counters, "no request counters were recorded"

        # Matched routes must carry their template. "unmatched" is correct for
        # a 404, which genuinely has no route, so the assertion targets the
        # routes we actually called rather than banning the label outright.
        assert any('endpoint="/health"' in line for line in counters), (
            "GET /api/v1/health is not labelled with its route template; the "
            "template is being resolved before routing has happened. "
            f"Observed: {counters}"
        )
        assert any('endpoint="/models"' in line for line in counters)

    async def test_path_parameters_do_not_inflate_cardinality(
        self, client: Any, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A million job lookups must share one time series.

        Labelling by raw path would create a series per job ID and eventually
        exhaust Prometheus's memory.
        """

        class _Unknown:
            state = "PENDING"
            info: dict[str, Any] = {}

            class backend:  # noqa: N801
                @staticmethod
                def get_task_meta(_: str) -> dict[str, Any]:
                    return {}

        monkeypatch.setattr("api.routers.batch.AsyncResult", lambda _: _Unknown())

        for job_id in ("job-a", "job-b", "job-c"):
            await client.get(f"/api/v1/batch/{job_id}", headers=auth_headers)

        body = (await client.get("/api/v1/metrics")).text
        batch_counters = [
            line
            for line in body.splitlines()
            if line.startswith("http_requests_total{") and "/batch/" in line
        ]

        for job_id in ("job-a", "job-b", "job-c"):
            assert not any(job_id in line for line in batch_counters), (
                f"{job_id} appears in a metric label; path parameters are not being templated"
            )
