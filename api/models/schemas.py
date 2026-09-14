"""Request schemas and shared enumerations.

Request models are deliberately strict: ``extra="forbid"`` means a typo in a
field name is a 422 with the offending key named, rather than a silently
ignored parameter and a caller wondering why their setting had no effect.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base for request bodies: reject unknown fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TaskType(StrEnum):
    """Inference task."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    EMBEDDING = "embedding"


class JobStatus(StrEnum):
    """Lifecycle of an asynchronous batch job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClassificationOptions(StrictModel):
    """Options for a classification request."""

    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of ranked predictions to return.",
    )
    model_version: str | None = Field(
        default=None,
        description="Pin a specific model version. Defaults to the active version.",
    )
    include_probabilities: bool = Field(
        default=True,
        description="Include softmax probabilities alongside labels.",
    )


class DetectionOptions(StrictModel):
    """Options for an object-detection request."""

    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Discard detections scoring below this threshold.",
    )
    max_detections: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Cap on returned detections, highest scoring first.",
    )
    model_version: str | None = None


class BatchItem(StrictModel):
    """One image in a batch request.

    Images are supplied as URLs rather than inline base64. Base64 inflates the
    payload by a third and forces the whole batch into memory at once; URLs let
    the worker fetch lazily and keep the submitting request small.
    """

    image_url: str = Field(description="HTTP(S) URL of the image to process.")
    item_id: str | None = Field(
        default=None,
        description="Caller-supplied identifier, echoed back in results.",
    )

    @field_validator("image_url")
    @classmethod
    def _require_http_scheme(cls, value: str) -> str:
        """Reject non-HTTP schemes.

        Without this, ``file://`` or ``gopher://`` URLs would let a caller read
        local files or reach internal services through the worker — a
        server-side request forgery. Network-level egress restrictions are the
        second layer; this is the first.
        """
        if not value.startswith(("http://", "https://")):
            raise ValueError("image_url must be an http:// or https:// URL")
        return value


class BatchRequest(StrictModel):
    """Submit images for asynchronous batch processing."""

    task: TaskType = Field(default=TaskType.CLASSIFICATION)
    items: list[BatchItem] = Field(min_length=1, description="Images to process.")
    classification_options: ClassificationOptions | None = None
    detection_options: DetectionOptions | None = None
    callback_url: str | None = Field(
        default=None,
        description="Optional URL notified with the result when the job finishes.",
    )

    @field_validator("callback_url")
    @classmethod
    def _require_http_scheme(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("callback_url must be an http:// or https:// URL")
        return value


class TokenRequest(StrictModel):
    """Exchange an API key for a short-lived access token."""

    api_key: str = Field(min_length=8, description="Issued API key.")
