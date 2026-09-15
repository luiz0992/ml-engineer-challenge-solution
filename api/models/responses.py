"""Response schemas.

Every inference response carries the ``model_name``, ``model_version``, and
``backend`` that produced it. That provenance is what makes a prediction
auditable after the fact: without it, a result recorded in a log or database
cannot be attributed to a particular artefact, and diagnosing a regression means
guessing which model was live at the time.

Timing is reported from the server's perspective (``inference_time_ms`` covers
preprocessing plus the forward pass) so clients can distinguish model latency
from network latency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from api.models.schemas import JobStatus, TaskType


class ResponseModel(BaseModel):
    """Base for responses.

    ``protected_namespaces=()`` disables Pydantic's warning about fields
    beginning with ``model_``. Those names are the clearest description of what
    the fields hold, and renaming them to satisfy a lint would make the public
    API worse.
    """

    model_config = ConfigDict(protected_namespaces=())


class ModelProvenance(ResponseModel):
    """Identifies the artefact that produced a result."""

    model_name: str
    model_version: str
    backend: str = Field(description="Runtime used, e.g. onnx, onnx-int8, tensorrt.")


class Prediction(ResponseModel):
    """One ranked classification result."""

    label: str = Field(description="Human-readable class name.")
    class_id: int = Field(description="Model output index.")
    wnid: str | None = Field(default=None, description="WordNet identifier.")
    probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Softmax probability."
    )


class ClassificationResponse(ResponseModel):
    """Result of a single-image classification."""

    # A worked example, so /docs shows a real payload rather than a
    # schema-generated placeholder of the right shape but implausible values.
    # Readers calibrate expectations from examples; "string" and 0 teach
    # nothing about what a probability distribution from this model looks like.
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "label": "tabby",
                        "class_id": 42,
                        "wnid": "n02123045",
                        "probability": 0.8731,
                    },
                    {
                        "label": "Egyptian cat",
                        "class_id": 43,
                        "wnid": "n02124075",
                        "probability": 0.0642,
                    },
                ],
                "inference_time_ms": 4.12,
                "correlation_id": "7b2f4c1e-9a3d-4f5b-8c6e-1d2a3b4c5d6e",
                "provenance": {
                    "model_name": "tiny-imagenet-classifier",
                    "model_version": "v1",
                    "backend": "onnx",
                },
                "cached": False,
            }
        },
    )

    predictions: list[Prediction]
    inference_time_ms: float = Field(description="Preprocessing plus forward pass.")
    correlation_id: str
    provenance: ModelProvenance
    cached: bool = Field(default=False, description="Whether this result was served from cache.")


class BoundingBox(ResponseModel):
    """Axis-aligned box in absolute pixel coordinates of the original image.

    Absolute rather than normalised coordinates: callers overlay these on the
    image they uploaded, and normalised values would require every client to
    know and reapply the original dimensions.
    """

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


class Detection(ResponseModel):
    """One detected object."""

    label: str
    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class DetectionResponse(ResponseModel):
    """Result of a single-image object detection.

    Boxes are in pixels of the image the caller uploaded, which the example
    makes concrete: `image_width`/`image_height` are the uploaded dimensions,
    not the 640x640 the model saw.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "detections": [
                    {
                        "label": "cat",
                        "class_id": 15,
                        "confidence": 0.9481,
                        "box": {
                            "x_min": 12.4,
                            "y_min": 54.9,
                            "x_max": 318.7,
                            "y_max": 472.1,
                        },
                    },
                    {
                        "label": "remote",
                        "class_id": 65,
                        "confidence": 0.7412,
                        "box": {
                            "x_min": 331.0,
                            "y_min": 402.8,
                            "x_max": 402.3,
                            "y_max": 448.2,
                        },
                    },
                ],
                "image_width": 640,
                "image_height": 480,
                "inference_time_ms": 9.87,
                "correlation_id": "7b2f4c1e-9a3d-4f5b-8c6e-1d2a3b4c5d6e",
                "provenance": {
                    "model_name": "rtdetr-coco-detector",
                    "model_version": "v1",
                    "backend": "onnx",
                },
                "cached": False,
            }
        },
    )

    detections: list[Detection]
    image_width: int
    image_height: int
    inference_time_ms: float
    correlation_id: str
    provenance: ModelProvenance
    cached: bool = False


class BatchItemResult(ResponseModel):
    """Outcome for one item in a batch.

    Carries either ``result`` or ``error``, never both. One bad image must not
    fail an otherwise successful batch, so failures are reported per item.
    """

    item_id: str | None = None
    image_url: str
    status: str = Field(description="'succeeded' or 'failed'.")
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class BatchJobResponse(ResponseModel):
    """Acknowledgement of a submitted batch job."""

    job_id: str
    status: JobStatus
    task: TaskType
    total_items: int
    submitted_at: datetime
    status_url: str = Field(description="Poll this URL for progress and results.")


class BatchJobStatusResponse(ResponseModel):
    """Progress and, once finished, results of a batch job."""

    job_id: str
    status: JobStatus
    task: TaskType
    total_items: int
    completed_items: int
    failed_items: int
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    results: list[BatchItemResult] | None = Field(
        default=None, description="Populated once the job reaches a terminal state."
    )
    error: str | None = None

    @property
    def progress(self) -> float:
        if self.total_items == 0:
            return 1.0
        return (self.completed_items + self.failed_items) / self.total_items


class SimilarImage(ResponseModel):
    """One retrieved neighbour."""

    rank: int = Field(description="1-based position, most similar first.")
    similarity: float = Field(ge=-1.0, le=1.0, description="Cosine similarity; 1.0 is identical.")
    label: str = Field(description="Class of the indexed image.")
    class_id: int
    reference: str = Field(
        description="Identifier of the indexed image, relative to the dataset root."
    )


class SimilarityResponse(ResponseModel):
    """Result of a similarity search."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "rank": 1,
                        "similarity": 0.8912,
                        "label": "tabby",
                        "class_id": 42,
                        "reference": "train/n02123045/images/n02123045_113.JPEG",
                    },
                    {
                        "rank": 2,
                        "similarity": 0.8514,
                        "label": "Egyptian cat",
                        "class_id": 43,
                        "reference": "train/n02124075/images/n02124075_9.JPEG",
                    },
                ],
                "index_size": 20000,
                "inference_time_ms": 3.41,
                "correlation_id": "7b2f4c1e-9a3d-4f5b-8c6e-1d2a3b4c5d6e",
                "provenance": {
                    "model_name": "tiny-imagenet-embedder",
                    "model_version": "v1",
                    "backend": "onnx",
                },
            }
        },
    )

    results: list[SimilarImage]
    index_size: int = Field(description="Number of images searched.")
    inference_time_ms: float
    correlation_id: str
    provenance: ModelProvenance


class ModelInfo(ResponseModel):
    """Metadata for one registered model version."""

    name: str
    version: str
    task: TaskType
    backend: str
    loaded: bool = Field(description="Whether the model is resident and ready.")
    is_active: bool = Field(description="Whether this version serves unpinned requests.")
    num_classes: int | None = None
    input_size: int | None = None
    artifact_size_mb: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    loaded_at: datetime | None = None


class ModelsResponse(ResponseModel):
    """All registered models."""

    models: list[ModelInfo]
    default_backend: str


class ComponentHealth(ResponseModel):
    """Health of one dependency."""

    name: str
    healthy: bool
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(ResponseModel):
    """Aggregate service health.

    ``status`` is ``healthy``, ``degraded``, or ``unhealthy``. The middle state
    matters operationally: if the cache is down but models still serve, the
    service is usable and should not be removed from the load balancer, but the
    condition must still be visible.
    """

    status: str
    version: str
    uptime_seconds: float
    components: list[ComponentHealth]


class TokenResponse(ResponseModel):
    """An issued access token."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth2 token type, not a secret
    expires_in: int = Field(description="Token lifetime in seconds.")
    tier: str


class ErrorDetail(ResponseModel):
    """Body of an error response."""

    code: str
    message: str
    details: dict[str, Any] | None = None
    correlation_id: str | None = None


class ErrorResponse(ResponseModel):
    """Envelope returned for every error.

    Every failure in the API renders through this one shape, including
    Pydantic validation errors, which FastAPI would otherwise return in a
    different format and force clients to parse twice.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "Rate limit of 10 requests per minute exceeded for tier 'free'.",
                    "details": {"limit": 10, "window_seconds": 60, "retry_after": 42},
                    "correlation_id": "7b2f4c1e-9a3d-4f5b-8c6e-1d2a3b4c5d6e",
                }
            }
        },
    )

    error: ErrorDetail
