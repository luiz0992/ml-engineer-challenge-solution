"""Request correlation and Prometheus metrics middleware.

Two ASGI middlewares:

* :class:`CorrelationMiddleware` assigns (or adopts) a correlation ID for every
  request, binds it to the logging context, and echoes it in the response.
* :class:`MetricsMiddleware` records request counts, latency, and in-flight
  gauges.

Metric label cardinality is the thing to get right. Labelling by raw URL path
would create a new time series per distinct path, and any endpoint containing an
identifier — ``/jobs/{job_id}`` — would grow the series count without bound and
eventually exhaust Prometheus's memory. The route *template* is used instead, so
a million job lookups share one series.
"""

from __future__ import annotations

import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from api.logging_config import bind_correlation_id, get_logger, new_correlation_id

logger = get_logger(__name__)

CORRELATION_HEADER = "X-Correlation-ID"

#: A dedicated registry rather than the global default, so tests can build and
#: discard applications without duplicate-timeseries errors.
REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["method", "endpoint", "status_code"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "endpoint"],
    # Buckets chosen around this service's actual latency profile: single-image
    # inference is ~1-20 ms, so the default Prometheus buckets (which start at
    # 5 ms and jump to 10 s) would put nearly every request in one bucket and
    # make percentile estimates useless.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Requests currently being served.",
    ["method", "endpoint"],
    registry=REGISTRY,
)

INFERENCE_COUNT = Counter(
    "inference_requests_total",
    "Inference requests by model and outcome.",
    ["model", "version", "backend", "outcome"],
    registry=REGISTRY,
)

INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Model inference latency, excluding HTTP overhead.",
    ["model", "version", "backend"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)

CACHE_OPERATIONS = Counter(
    "cache_operations_total",
    "Cache operations by result.",
    ["operation", "result"],
    registry=REGISTRY,
)

MODEL_INFO = Gauge(
    "model_loaded_info",
    "Loaded models. Value is 1 when resident.",
    ["model", "version", "backend", "degraded"],
    registry=REGISTRY,
)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request.

    An inbound ``X-Correlation-ID`` is adopted so a trace started upstream
    survives into this service's logs; otherwise one is generated. The value is
    always echoed back, giving clients something to quote in a bug report.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(CORRELATION_HEADER)
        # Bound to a sane length: the value reaches logs and headers, and an
        # unbounded client-supplied string is a log-injection vector.
        correlation_id = incoming[:64] if incoming and incoming.strip() else new_correlation_id()

        bind_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers[CORRELATION_HEADER] = correlation_id
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request counts and latency."""

    def __init__(self, app: ASGIApp, *, exclude_paths: frozenset[str] = frozenset()) -> None:
        super().__init__(app)
        # The metrics endpoint itself is excluded: scraping it would otherwise
        # inflate the counters it reports.
        self.exclude_paths = exclude_paths

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        endpoint = _route_template(request)
        method = request.method

        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        started = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - started
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            REQUEST_COUNT.labels(
                method=method, endpoint=endpoint, status_code=str(status_code)
            ).inc()

            logger.info(
                "request_complete",
                method=method,
                path=request.url.path,
                endpoint=endpoint,
                status_code=status_code,
                duration_ms=round(duration * 1000, 2),
            )


def _route_template(request: Request) -> str:
    """Return the matched route template, not the concrete path.

    Starlette populates ``request.scope["route"]`` after routing. Falling back
    to a constant for unmatched paths prevents 404 scans from creating a
    time series per probed URL.
    """
    route = request.scope.get("route")
    path_format = getattr(route, "path_format", None) or getattr(route, "path", None)
    return path_format or "unmatched"


def record_inference(
    *, model: str, version: str, backend: str, duration_seconds: float, success: bool
) -> None:
    """Record one inference for the metrics endpoint."""
    INFERENCE_COUNT.labels(
        model=model,
        version=version,
        backend=backend,
        outcome="success" if success else "failure",
    ).inc()
    if success:
        INFERENCE_LATENCY.labels(model=model, version=version, backend=backend).observe(
            duration_seconds
        )


def record_cache(operation: str, result: str) -> None:
    CACHE_OPERATIONS.labels(operation=operation, result=result).inc()


def record_model_loaded(*, model: str, version: str, backend: str, degraded: bool) -> None:
    MODEL_INFO.labels(
        model=model, version=version, backend=backend, degraded=str(degraded).lower()
    ).set(1)
