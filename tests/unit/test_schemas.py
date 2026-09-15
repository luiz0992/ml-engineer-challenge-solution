"""Tests for request and response schemas.

Schema validation is the first line of defence at the HTTP boundary. Two
behaviours matter beyond type checking: unknown fields are rejected rather than
ignored, and URL fields reject non-HTTP schemes, which is what prevents a
caller from reaching internal services or local files through the worker.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from api.models.responses import (
    BatchJobStatusResponse,
    BoundingBox,
    ClassificationResponse,
    DetectionResponse,
    ErrorResponse,
    SimilarityResponse,
)
from api.models.schemas import (
    BatchItem,
    BatchRequest,
    ClassificationOptions,
    DetectionOptions,
    JobStatus,
    TaskType,
    TokenRequest,
)

pytestmark = pytest.mark.unit


class TestStrictness:
    def test_unknown_fields_are_rejected(self) -> None:
        """A typo must be an error, not a silently ignored setting.

        Without this a caller sets `top_kk=10`, receives 5 predictions, and has
        no way to discover why.
        """
        with pytest.raises(ValidationError) as exc_info:
            # The typo is deliberate; mypy flags it, which is the behaviour
            # under test at runtime too.
            ClassificationOptions(top_k=5, tpo_k=10)  # type: ignore[call-arg]

        assert "tpo_k" in str(exc_info.value)

    def test_defaults_are_applied(self) -> None:
        options = ClassificationOptions()
        assert options.top_k == 5
        assert options.include_probabilities is True
        assert options.model_version is None


class TestClassificationOptions:
    @pytest.mark.parametrize("top_k", [0, -1, 101])
    def test_rejects_out_of_range_top_k(self, top_k: int) -> None:
        with pytest.raises(ValidationError):
            ClassificationOptions(top_k=top_k)

    @pytest.mark.parametrize("top_k", [1, 5, 100])
    def test_accepts_in_range_top_k(self, top_k: int) -> None:
        assert ClassificationOptions(top_k=top_k).top_k == top_k


class TestDetectionOptions:
    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_rejects_threshold_outside_probability_range(self, threshold: float) -> None:
        with pytest.raises(ValidationError):
            DetectionOptions(confidence_threshold=threshold)

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_accepts_valid_threshold(self, threshold: float) -> None:
        assert DetectionOptions(confidence_threshold=threshold).confidence_threshold == threshold


class TestBatchItemURLValidation:
    @pytest.mark.parametrize("url", ["http://example.com/a.jpg", "https://example.com/a.jpg"])
    def test_accepts_http_urls(self, url: str) -> None:
        assert BatchItem(image_url=url).image_url == url

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "gopher://internal.service/",
            "ftp://example.com/a.jpg",
            "//example.com/a.jpg",
            "/etc/passwd",
            "data:image/png;base64,iVBORw0KGgo=",
        ],
    )
    def test_rejects_non_http_schemes(self, url: str) -> None:
        """Server-side request forgery is blocked at the schema boundary.

        Without this, a caller could make the worker read local files or reach
        services that are only routable from inside the network. Network egress
        rules are the second layer; this is the first.
        """
        with pytest.raises(ValidationError):
            BatchItem(image_url=url)

    def test_item_id_is_optional_and_echoed_back(self) -> None:
        assert BatchItem(image_url="https://e.com/a.jpg", item_id="my-id").item_id == "my-id"
        assert BatchItem(image_url="https://e.com/a.jpg").item_id is None


class TestBatchRequest:
    def test_requires_at_least_one_item(self) -> None:
        with pytest.raises(ValidationError):
            BatchRequest(items=[])

    def test_defaults_to_classification(self) -> None:
        request = BatchRequest(items=[BatchItem(image_url="https://e.com/a.jpg")])
        assert request.task is TaskType.CLASSIFICATION

    def test_rejects_non_http_callback_url(self) -> None:
        """The callback is fetched by the worker and carries the same risk."""
        with pytest.raises(ValidationError):
            BatchRequest(
                items=[BatchItem(image_url="https://e.com/a.jpg")],
                callback_url="file:///tmp/x",
            )

    def test_accepts_http_callback_url(self) -> None:
        request = BatchRequest(
            items=[BatchItem(image_url="https://e.com/a.jpg")],
            callback_url="https://hooks.example.com/done",
        )
        assert request.callback_url == "https://hooks.example.com/done"


class TestTokenRequest:
    def test_rejects_implausibly_short_key(self) -> None:
        with pytest.raises(ValidationError):
            TokenRequest(api_key="short")

    def test_accepts_a_realistic_key(self) -> None:
        assert TokenRequest(api_key="dev-key-pro").api_key == "dev-key-pro"


class TestResponseHelpers:
    def test_bounding_box_derives_dimensions(self) -> None:
        box = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=170)
        assert box.width == 100
        assert box.height == 150

    def test_bounding_box_rejects_negative_coordinates(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x_min=-1, y_min=0, x_max=10, y_max=10)

    def test_job_progress_is_a_fraction(self) -> None:
        from datetime import UTC, datetime

        response = BatchJobStatusResponse(
            job_id="j",
            status=JobStatus.RUNNING,
            task=TaskType.CLASSIFICATION,
            total_items=10,
            completed_items=6,
            failed_items=2,
            submitted_at=datetime.now(UTC),
        )
        # Failures count as finished: a job with 8 of 10 resolved is 80% done
        # regardless of how those 8 turned out.
        assert response.progress == pytest.approx(0.8)

    def test_progress_of_an_empty_job_is_complete(self) -> None:
        """Guards against a division by zero on a degenerate job."""
        from datetime import UTC, datetime

        response = BatchJobStatusResponse(
            job_id="j",
            status=JobStatus.COMPLETED,
            task=TaskType.CLASSIFICATION,
            total_items=0,
            completed_items=0,
            failed_items=0,
            submitted_at=datetime.now(UTC),
        )
        assert response.progress == 1.0


class TestDocumentedExamples:
    """The examples rendered in /docs must be valid instances of their models.

    An example is copied by readers and pasted into clients, so a wrong field
    name there is worse than no example at all -- it is documentation that
    actively misleads. Writing these by hand produced exactly that mistake
    (`score` for what is really `similarity`), which is why it is checked
    rather than reviewed.
    """

    @pytest.mark.parametrize(
        "model",
        [
            ClassificationResponse,
            DetectionResponse,
            SimilarityResponse,
            ErrorResponse,
        ],
    )
    def test_example_validates_against_its_own_schema(self, model: type[BaseModel]) -> None:
        example = model.model_json_schema().get("example")

        assert example is not None, f"{model.__name__} documents no example"
        model.model_validate(example)
