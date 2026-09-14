"""Structured logging with request correlation.

Every log line carries a ``correlation_id`` that ties it to the originating
request, which is what makes logs usable when several requests are in flight
concurrently. Without it, interleaved async handlers produce output that cannot
be attributed to any particular request.

The correlation ID lives in a :class:`~contextvars.ContextVar` rather than being
threaded through call signatures. ContextVars propagate correctly across
``await`` boundaries and are isolated per task, so a service function deep in
the call stack can log with full context without every intermediate function
having to accept and forward an ID it does not otherwise use.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

#: Set by the correlation middleware at the start of each request.
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

#: Set by the auth middleware once the caller is identified. Logging the user
#: alongside the request makes per-tenant debugging and abuse investigation
#: possible without joining against another data source.
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)


def new_correlation_id() -> str:
    """Generate a correlation ID for a request."""
    return uuid.uuid4().hex


def bind_correlation_id(correlation_id: str) -> None:
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> str | None:
    return correlation_id_var.get()


def bind_user_id(user_id: str | None) -> None:
    user_id_var.set(user_id)


def _add_request_context(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor injecting the current request context."""
    if (correlation_id := correlation_id_var.get()) is not None:
        event_dict["correlation_id"] = correlation_id
    if (user_id := user_id_var.get()) is not None:
        event_dict["user_id"] = user_id
    return event_dict


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog and route the stdlib logging tree through it.

    Third-party libraries (uvicorn, SQLAlchemy, ONNX Runtime) log through the
    standard library. Routing those records through the same structlog pipeline
    keeps the output format uniform, so a log aggregator sees one schema rather
    than a mix of JSON and free text.

    JSON is the default because production logs are consumed by machines. The
    console renderer is for local development, where a human is reading them.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_request_context,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Replace existing handlers so repeated configuration (common under
    # uvicorn's reloader and in tests) does not duplicate every log line.
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn installs its own handlers; clearing them and letting records
    # propagate to root avoids each access log appearing twice in two formats.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger."""
    return structlog.stdlib.get_logger(name)
