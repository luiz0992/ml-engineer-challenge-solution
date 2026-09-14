"""Custom exception hierarchy and error responses.

All application errors derive from :class:`APIError`, which carries an HTTP
status, a stable machine-readable ``code``, and a human-readable message. The
code is what clients should branch on: HTTP status alone is too coarse
(a 400 could be an unsupported format, an oversized file, or a corrupt image)
and message text is not a stable contract.

Errors never leak internal detail. A client learns what it did wrong and what
to do about it; stack traces, file paths, and library internals go to the logs
under the request's correlation ID, which is returned so a user can quote it in
a support request.
"""

from __future__ import annotations

from typing import Any


class APIError(Exception):
    """Base class for application errors.

    Attributes:
        status_code: HTTP status to return.
        code: Stable identifier clients can branch on.
        message: Human-readable explanation, safe to show a caller.
        details: Optional structured context, also caller-safe.
    """

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An internal error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or {}
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self, correlation_id: str | None = None) -> dict[str, Any]:
        """Render as the JSON body returned to the client."""
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.details:
            payload["error"]["details"] = self.details
        if correlation_id:
            payload["error"]["correlation_id"] = correlation_id
        return payload


# --- 4xx: caller errors ----------------------------------------------------
class ValidationError(APIError):
    """The request was structurally valid but semantically unacceptable."""

    status_code = 422
    code = "validation_error"
    message = "The request could not be processed."


class InvalidImageError(ValidationError):
    """The uploaded bytes are not a usable image."""

    code = "invalid_image"
    message = "The uploaded file is not a valid image."


class UnsupportedFormatError(ValidationError):
    """The image format is not accepted."""

    code = "unsupported_format"
    message = "The image format is not supported."


class PayloadTooLargeError(APIError):
    """The upload exceeds the configured size limit."""

    status_code = 413
    code = "payload_too_large"
    message = "The uploaded file is too large."


class BatchTooLargeError(ValidationError):
    """More items were submitted than the batch endpoint accepts."""

    code = "batch_too_large"
    message = "The batch exceeds the maximum permitted size."


class AuthenticationError(APIError):
    """No credentials, or credentials that could not be verified."""

    status_code = 401
    code = "authentication_required"
    message = "Valid authentication credentials are required."


class AuthorizationError(APIError):
    """Authenticated, but not permitted to perform this action."""

    status_code = 403
    code = "forbidden"
    message = "You do not have permission to perform this action."


class RateLimitExceededError(APIError):
    """The caller exceeded the request quota for their tier."""

    status_code = 429
    code = "rate_limit_exceeded"
    message = "Rate limit exceeded. Please retry later."

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after_seconds: int,
        limit: int,
        **kwargs: Any,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        details = {"limit_per_minute": limit, "retry_after_seconds": retry_after_seconds}
        details.update(kwargs.pop("details", {}))
        super().__init__(message, details=details, **kwargs)


class ModelNotFoundError(APIError):
    """The requested model or version is not registered."""

    status_code = 404
    code = "model_not_found"
    message = "The requested model is not available."


class JobNotFoundError(APIError):
    """The requested background job does not exist or has expired."""

    status_code = 404
    code = "job_not_found"
    message = "The requested job was not found."


# --- 5xx: server errors ----------------------------------------------------
class InferenceError(APIError):
    """The model failed to produce a prediction."""

    status_code = 500
    code = "inference_failed"
    message = "Inference failed. The request was not processed."


class ModelUnavailableError(APIError):
    """No usable backend could serve the request.

    Distinct from :class:`InferenceError`: this means the service is degraded
    rather than that one request failed, so it returns 503 and is retryable.
    """

    status_code = 503
    code = "model_unavailable"
    message = "The model is temporarily unavailable. Please retry."


class ServiceUnavailableError(APIError):
    """A required dependency (cache, database, broker) is unreachable."""

    status_code = 503
    code = "service_unavailable"
    message = "A required service is temporarily unavailable."
