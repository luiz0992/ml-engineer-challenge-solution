"""Load and stress profiles for the serving API.

Complements the micro-benchmarks in `models/optimisation/benchmark.py`, which
time the model in isolation. This measures the whole path a client actually
experiences: TLS termination at the gateway, authentication, rate limiting,
cache lookup, preprocessing, inference, and the audit write.

Those numbers diverge, and the gap is the interesting part. The classifier runs
in 1.06 ms under TensorRT; if p95 at the edge is 40 ms, the model is not the
bottleneck and optimising it further is wasted effort.

Run against a stack started with `docker compose up -d`::

    # Interactive, with the web UI on :8089
    uv run locust -f tests/performance/locustfile.py --host http://localhost:8080

    # Headless: 50 users, ramping 5/s, for two minutes
    uv run locust -f tests/performance/locustfile.py --host http://localhost:8080 \\
        --headless --users 50 --spawn-rate 5 --run-time 2m \\
        --html benchmarks/load-test.html

    # Stress the rate limiter: ramp the free tier until it saturates
    uv run locust -f tests/performance/locustfile.py --host http://localhost:8080 \\
        --headless --users 500 --spawn-rate 10 --run-time 5m FreeTierUser

Naming user classes on the command line selects them; omitting the name runs
all three. `FreeTierUser` exists to exercise the 429 path under load, which is
a supported outcome rather than a failure.
"""

from __future__ import annotations

import io
import random
import time
from typing import Any

from locust import HttpUser, between, constant, events, task
from PIL import Image

#: Development keys, seeded by the API in non-production environments only.
#: There is nothing to leak: `_seed_api_keys` refuses to create these when
#: APP_ENV is production.
API_KEYS = {
    "free": "dev-key-free",
    "pro": "dev-key-pro",
    "enterprise": "dev-key-enterprise",
}

#: A pool of pre-encoded images, built once at startup. Encoding a JPEG per
#: request would put PIL on the hot path and measure the load generator rather
#: than the service -- a classic way to produce a flat throughput ceiling that
#: has nothing to do with the system under test.
_IMAGE_POOL: list[bytes] = []


@events.test_start.add_listener
def build_image_pool(**_: Any) -> None:
    """Pre-encode the request bodies before the run begins."""
    _IMAGE_POOL.clear()
    rng = random.Random(0)
    for _index in range(32):
        size = rng.choice([224, 256, 384, 512])
        image = Image.new(
            "RGB",
            (size, size),
            (rng.randrange(256), rng.randrange(256), rng.randrange(256)),
        )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        _IMAGE_POOL.append(buffer.getvalue())


def _image() -> bytes:
    return random.choice(_IMAGE_POOL)


class _AuthenticatedUser(HttpUser):
    """Shared token handling.

    Abstract, so Locust does not instantiate it directly.
    """

    abstract = True

    #: Which seeded key this user authenticates with.
    tier = "pro"

    #: Attempts to obtain a token before giving up. The gateway rate-limits
    #: `/auth/token` far more tightly than inference, because it is the natural
    #: target for credential stuffing -- so a spawn burst legitimately gets
    #: 429s that a real client would simply retry.
    AUTH_ATTEMPTS = 6

    def on_start(self) -> None:
        """Exchange the API key for a bearer token once per simulated user.

        Tokens are short-lived but outlast a typical run. Re-authenticating per
        request would measure the token endpoint instead of inference.

        A 429 here is retried rather than failed. Spawning 20 users at once
        trips the gateway's auth limit by design, and treating that as a test
        failure would report the load generator's impatience as a defect in the
        service.
        """
        for attempt in range(self.AUTH_ATTEMPTS):
            with self.client.post(
                "/api/v1/auth/token",
                json={"api_key": API_KEYS[self.tier]},
                name="POST /auth/token",
                catch_response=True,
            ) as response:
                if response.status_code == 200:
                    response.success()
                    self.client.headers["Authorization"] = (
                        f"Bearer {response.json()['access_token']}"
                    )
                    return

                if response.status_code == 429:
                    response.success()
                    # Exponential backoff with jitter, so retrying users do not
                    # re-synchronise into the next burst.
                    time.sleep(min(2**attempt, 10) * (0.5 + random.random()))
                    continue

                response.failure(f"Could not authenticate: {response.status_code}")
                self.stop()
                return

        # Out of attempts. Stop this user, not the run: `runner.quit()` here
        # turns one transient 502 during a rolling restart into a terminated
        # test, and at the documented stress settings the gateway's auth limit
        # (5 r/s) is below the spawn rate, so exhausting retries is expected
        # for some users rather than a defect in the service.
        self.stop()


class ClassificationUser(_AuthenticatedUser):
    """The dominant traffic pattern: single-image classification.

    `between(0.5, 2)` models think time. Zero wait would generate a closed-loop
    hammer whose latency is dominated by queueing at the client, which is not
    what a real client population looks like.
    """

    # Enterprise, not pro. Every simulated user authenticates with the same
    # seeded key, so they share one rate-limit bucket: 20 users at ~1.3 s think
    # time is roughly 15 req/s, which fits inside enterprise's 1200/min but
    # would spend most of a run being 429'd on pro's 120/min. Driving past
    # ~20 req/s needs per-user API keys rather than a bigger `--users`.
    tier = "enterprise"
    wait_time = between(0.5, 2.0)

    @task(10)
    def classify(self) -> None:
        """Measure the uncached inference path.

        `use_cache=false` is essential, not incidental. The image pool holds 32
        entries, so without it everything after the first 32 requests is a cache
        hit and this task reports the cache's latency under the name of the
        model's -- which is exactly the mistake the separate cached task below
        exists to keep visible.
        """
        self.client.post(
            "/api/v1/classify",
            files={"file": ("load.jpg", _image(), "image/jpeg")},
            params={"use_cache": "false"},
            name="POST /classify",
        )

    @task(2)
    def classify_cache_hit(self) -> None:
        """Repeat one fixed image, so this path should hit the result cache.

        Named separately because mixing cached and uncached responses into one
        statistic produces a bimodal distribution whose percentiles describe
        neither case.
        """
        self.client.post(
            "/api/v1/classify",
            files={"file": ("cached.jpg", _IMAGE_POOL[0], "image/jpeg")},
            name="POST /classify (cached)",
        )

    @task(3)
    def detect(self) -> None:
        with self.client.post(
            "/api/v1/detect",
            files={"file": ("load.jpg", _image(), "image/jpeg")},
            params={"use_cache": "false"},
            name="POST /detect",
            catch_response=True,
        ) as response:
            # A deployment may serve classification only; the detector degrades
            # to 503 by design rather than failing startup.
            if response.status_code == 503:
                response.success()

    @task(1)
    def list_models(self) -> None:
        self.client.get("/api/v1/models", name="GET /models")

    @task(1)
    def health(self) -> None:
        self.client.get("/api/v1/health", name="GET /health")


class BatchUser(_AuthenticatedUser):
    """Submits background jobs and polls them.

    Submission should stay fast regardless of batch size -- the endpoint
    enqueues and returns 202. A submission latency that grows with item count
    means work leaked onto the request path.
    """

    tier = "enterprise"
    wait_time = between(5.0, 15.0)

    @task
    def submit_and_poll(self) -> None:
        items = [
            {"image_url": f"https://example.invalid/{i}.jpg", "item_id": str(i)}
            for i in range(random.randint(2, 10))
        ]

        with self.client.post(
            "/api/v1/batch",
            json={"task": "classification", "items": items},
            name="POST /batch",
            catch_response=True,
        ) as response:
            if response.status_code != 202:
                return
            response.success()
            job_id = response.json()["job_id"]

        self.client.get(f"/api/v1/batch/{job_id}", name="GET /batch/{job_id}")


class FreeTierUser(_AuthenticatedUser):
    """Hammers the API on the lowest tier to exercise rate limiting under load.

    The point is not throughput. It is that rejection stays cheap and correct
    while the limiter is saturated: a 429 must arrive fast, carry `Retry-After`,
    and never consume model capacity. A limiter that becomes slow under
    contention converts a protection mechanism into an outage.
    """

    tier = "free"
    wait_time = constant(0)

    @task
    def classify_until_limited(self) -> None:
        with self.client.post(
            "/api/v1/classify",
            files={"file": ("stress.jpg", _image(), "image/jpeg")},
            params={"use_cache": "false"},
            name="POST /classify (free tier)",
            catch_response=True,
        ) as response:
            if response.status_code == 429:
                # The expected outcome, not an error. Counting it as a failure
                # would report a 90% failure rate for a limiter working exactly
                # as designed.
                if "Retry-After" not in response.headers:
                    response.failure("429 without a Retry-After header")
                else:
                    response.success()


@events.quitting.add_listener
def enforce_thresholds(environment: Any, **_: Any) -> None:
    """Fail the run on a bad result, so CI can gate on it.

    Locust exits 0 by default no matter how the run went, which makes it
    useless as a check. These thresholds are deliberately loose -- they catch a
    service that is broken or badly degraded, not a few percent of drift.
    """
    stats = environment.stats.total

    if stats.num_requests == 0:
        environment.process_exit_code = 1
        return

    failure_ratio = stats.fail_ratio
    p95 = stats.get_response_time_percentile(0.95)

    degraded = failure_ratio > 0.01 or (p95 is not None and p95 > 2000)
    environment.process_exit_code = 1 if degraded else 0
