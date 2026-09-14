"""Tests for object detection post-processing.

Detection is most often silently wrong in post-processing rather than in the
model, so these tests target the coordinate transform and the filtering logic
with constructed inputs whose correct output is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.utils.image_processing import (
    DETECTION_IMAGE_SIZE,
    boxes_to_absolute,
    preprocess_for_detection,
)
from tests.conftest import make_image_bytes

pytestmark = pytest.mark.unit


class TestDetectionPreprocessing:
    def test_produces_the_fixed_input_size(self) -> None:
        """RT-DETR has learned positional priors at 640x640; it is not tunable."""
        array, _, _ = preprocess_for_detection(make_image_bytes(800, 600))

        assert array.shape == (3, DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE)
        assert array.dtype == np.float32

    def test_returns_original_dimensions(self) -> None:
        """Needed to map normalised boxes back to the caller's pixel space."""
        _, width, height = preprocess_for_detection(make_image_bytes(800, 600))

        assert (width, height) == (800, 600)

    def test_scales_to_unit_range_without_normalising(self) -> None:
        """RT-DETR sets do_normalize=False; only a 1/255 rescale is applied.

        Applying ImageNet statistics here by analogy with the classifier would
        shift the input distribution and degrade detection with no error. A
        normalised array would contain negative values; this one must not.
        """
        array, _, _ = preprocess_for_detection(make_image_bytes(256, 256))

        assert array.min() >= 0.0
        assert array.max() <= 1.0

    def test_resize_does_not_preserve_aspect_ratio(self) -> None:
        """The processor squashes to a square with do_pad=False.

        Letterboxing instead would place objects where the model does not
        expect them, and would make the inverse box transform wrong.
        """
        array, _, _ = preprocess_for_detection(make_image_bytes(1000, 200))

        assert array.shape[1] == array.shape[2] == DETECTION_IMAGE_SIZE

    @pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
    def test_handles_non_rgb_input(self, mode: str) -> None:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new(mode, (320, 240)).save(buffer, format="PNG")

        array, _, _ = preprocess_for_detection(buffer.getvalue())
        assert array.shape[0] == 3


class TestBoxTransform:
    def test_centre_format_converts_to_corners(self) -> None:
        """A box covering the middle half of a 200x100 image."""
        boxes = np.array([[0.5, 0.5, 0.5, 0.5]])

        result = boxes_to_absolute(boxes, 200, 100)

        assert result[0] == pytest.approx([50.0, 25.0, 150.0, 75.0])

    def test_full_image_box(self) -> None:
        result = boxes_to_absolute(np.array([[0.5, 0.5, 1.0, 1.0]]), 640, 480)
        assert result[0] == pytest.approx([0.0, 0.0, 640.0, 480.0])

    def test_axes_scale_independently(self) -> None:
        """Because the resize squashed the image, each axis inverts separately.

        A shared scale factor -- the natural mistake if letterboxing were
        assumed -- would distort every box on a non-square image.
        """
        result = boxes_to_absolute(np.array([[0.25, 0.75, 0.5, 0.5]]), 400, 100)

        assert result[0][0] == pytest.approx(0.0)  # (0.25 - 0.25) * 400
        assert result[0][2] == pytest.approx(200.0)  # (0.25 + 0.25) * 400
        assert result[0][1] == pytest.approx(50.0)  # (0.75 - 0.25) * 100
        assert result[0][3] == pytest.approx(100.0)  # (0.75 + 0.25) * 100

    def test_boxes_are_clipped_to_the_image(self) -> None:
        """A predicted box may extend past the edge; a reported one may not.

        Clients draw these directly, and a negative or oversized coordinate
        either throws or renders off-canvas.
        """
        boxes = np.array([[0.5, 0.5, 2.0, 2.0]])  # twice the image in each axis

        result = boxes_to_absolute(boxes, 640, 480)

        assert result[0][0] >= 0.0
        assert result[0][1] >= 0.0
        assert result[0][2] <= 640.0
        assert result[0][3] <= 480.0

    def test_handles_a_batch_of_boxes(self) -> None:
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]])
        assert boxes_to_absolute(boxes, 100, 100).shape == (2, 4)


class TestDetectionDecoding:
    """The filtering and ranking applied to raw query outputs."""

    @staticmethod
    def _decode(scores: np.ndarray, boxes: np.ndarray, **kwargs):
        from types import SimpleNamespace

        from api.services.detection_service import DetectionService

        model = SimpleNamespace(class_names=[f"class_{i}" for i in range(4)])
        params = {"threshold": 0.5, "max_detections": 100, "width": 100, "height": 100}
        params.update(kwargs)
        return DetectionService._decode(
            scores,
            boxes,
            model,  # type: ignore[arg-type]
            params["width"],
            params["height"],
            params["threshold"],
            params["max_detections"],
        )

    def test_discards_low_confidence_queries(self) -> None:
        scores = np.array([[0.9, 0.1, 0.0, 0.0], [0.2, 0.1, 0.0, 0.0]])
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]])

        detections = self._decode(scores, boxes, threshold=0.5)

        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.9)

    def test_each_query_yields_at_most_one_detection(self) -> None:
        """Emitting every class above threshold would double-report one object."""
        scores = np.array([[0.9, 0.8, 0.7, 0.6]])
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]])

        detections = self._decode(scores, boxes, threshold=0.5)

        assert len(detections) == 1
        assert detections[0].class_id == 0

    def test_results_are_ranked_by_confidence(self) -> None:
        scores = np.array([[0.6, 0, 0, 0], [0.9, 0, 0, 0], [0.7, 0, 0, 0]])
        boxes = np.tile(np.array([0.5, 0.5, 0.2, 0.2]), (3, 1))

        detections = self._decode(scores, boxes, threshold=0.5)

        assert [d.confidence for d in detections] == pytest.approx([0.9, 0.7, 0.6])

    def test_cap_keeps_the_most_confident(self) -> None:
        """Truncation must follow ranking, not precede it.

        Slicing the raw 300 queries first would return an arbitrary subset and
        discard the best detections.
        """
        scores = np.zeros((10, 4))
        scores[:, 0] = np.linspace(0.51, 0.99, 10)
        boxes = np.tile(np.array([0.5, 0.5, 0.2, 0.2]), (10, 1))

        detections = self._decode(scores, boxes, threshold=0.5, max_detections=3)

        assert len(detections) == 3
        assert detections[0].confidence == pytest.approx(0.99)

    def test_returns_empty_when_nothing_is_confident(self) -> None:
        scores = np.full((5, 4), 0.1)
        boxes = np.tile(np.array([0.5, 0.5, 0.2, 0.2]), (5, 1))

        assert self._decode(scores, boxes, threshold=0.5) == []

    def test_labels_resolve_from_the_class_map(self) -> None:
        scores = np.array([[0.0, 0.0, 0.95, 0.0]])
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]])

        detections = self._decode(scores, boxes, threshold=0.5)

        assert detections[0].label == "class_2"
        assert detections[0].class_id == 2
