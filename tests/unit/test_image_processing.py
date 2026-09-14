"""Tests for inference preprocessing.

The central test here is :class:`TestTorchvisionEquivalence`, which asserts the
serving pipeline matches the training-time evaluation transform numerically.

That test exists because two real bugs were found this way during development,
both of which raise no error and produce no visible defect:

1. ``v2.Resize`` defaults to BILINEAR, not BICUBIC. Using bicubic shifted
   activations by up to 2.5 in normalised units.
2. ``CenterCrop`` offsets are ``round((dim - size) / 2)``, not floor division.
   For a 275-wide image these differ by one pixel.

Either would have degraded accuracy silently. Since the serving path
deliberately reimplements preprocessing in NumPy and Pillow — so the production
image needs no torch — this equivalence is the only thing preventing the two
implementations from drifting apart again.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.utils.image_processing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessConfig,
    center_crop,
    decode_image,
    preprocess_image,
    resize_short_edge,
    softmax,
    stack_batch,
    top_k,
)
from tests.conftest import make_image_bytes

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch", reason="equivalence tests require torchvision")


class TestDecode:
    @pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P", "CMYK"])
    def test_always_produces_rgb(self, mode: str) -> None:
        """Every colour mode is normalised to RGB.

        Without this the model receives the wrong channel count and fails deep
        inside the graph with an unhelpful shape error.
        """
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new(mode, (64, 64)).save(buffer, format="TIFF" if mode == "CMYK" else "PNG")

        assert decode_image(buffer.getvalue()).mode == "RGB"


class TestResize:
    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [
            (512, 256, (512, 256)),  # short edge is height
            (256, 512, (256, 512)),  # short edge is width
            (256, 256, (256, 256)),  # already square
        ],
    )
    def test_short_edge_hits_target_and_ratio_is_preserved(
        self, width: int, height: int, expected: tuple[int, int]
    ) -> None:
        from PIL import Image

        resized = resize_short_edge(Image.new("RGB", (width, height)), 256)
        assert min(resized.size) == 256

        original_ratio = width / height
        new_ratio = resized.size[0] / resized.size[1]
        assert new_ratio == pytest.approx(original_ratio, rel=0.01)


class TestCenterCrop:
    def test_uses_rounding_not_floor_division(self) -> None:
        """Offsets must round, matching torchvision.

        For a 275-wide image cropped to 224, ``round(25.5) = 26`` while
        ``51 // 2 = 25``. This is the one-pixel bug described in the module
        docstring: silent, and it shifts every activation slightly off the
        distribution the model was trained on.
        """
        from PIL import Image
        from torchvision.transforms import v2

        source = Image.fromarray(
            np.random.default_rng(0).integers(0, 255, (256, 275, 3), dtype=np.uint8)
        )

        ours = np.asarray(center_crop(source, 224))
        theirs = np.asarray(v2.CenterCrop(224)(source))

        assert np.array_equal(ours, theirs), "center_crop diverged from torchvision"

    def test_output_is_square_and_correct_size(self) -> None:
        from PIL import Image

        assert center_crop(Image.new("RGB", (300, 200)), 128).size == (128, 128)


class TestTorchvisionEquivalence:
    """The serving pipeline must match the training evaluation transform."""

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            (64, 64),  # native Tiny-ImageNet resolution
            (300, 200),  # landscape
            (200, 300),  # portrait
            (513, 477),  # odd dimensions -> exercises crop rounding
            (275, 256),  # the exact case that exposed the off-by-one
            (1024, 768),  # large
            (256, 256),  # already at the resize target
        ],
    )
    def test_matches_training_transform(self, width: int, height: int) -> None:
        from PIL import Image

        from models.data.augmentation import AugmentationConfig, build_eval_transform

        image_bytes = make_image_bytes(width, height, fmt="PNG")

        ours = preprocess_image(image_bytes, PreprocessConfig(image_size=224))
        theirs = build_eval_transform(AugmentationConfig(image_size=224))(
            Image.open(__import__("io").BytesIO(image_bytes))
        ).numpy()

        assert ours.shape == theirs.shape
        max_diff = float(np.abs(ours - theirs).max())
        assert max_diff < 1e-4, (
            f"Serving preprocessing diverged from the training transform by "
            f"{max_diff:.3e} at {width}x{height}. This silently degrades accuracy; "
            f"check the resize interpolation mode and the centre-crop offset."
        )

    def test_uses_imagenet_statistics(self) -> None:
        """Normalisation constants must match the training configuration.

        Using Tiny-ImageNet's own statistics against an ImageNet-pretrained
        backbone is a plausible mistake that costs accuracy with no error.
        """
        from models.data.augmentation import IMAGENET_MEAN as TRAIN_MEAN
        from models.data.augmentation import IMAGENET_STD as TRAIN_STD

        assert IMAGENET_MEAN == TRAIN_MEAN
        assert IMAGENET_STD == TRAIN_STD


class TestPreprocessOutput:
    def test_returns_chw_float32(self) -> None:
        array = preprocess_image(make_image_bytes(), PreprocessConfig(image_size=224))
        assert array.shape == (3, 224, 224)
        assert array.dtype == np.float32

    def test_normalisation_centres_the_distribution(self) -> None:
        """Normalised output should be roughly zero-mean, not in [0, 1].

        Catches the common error of forgetting to normalise, which leaves
        values in [0, 1] and quietly degrades every prediction.
        """
        array = preprocess_image(make_image_bytes(seed=1), PreprocessConfig(image_size=224))
        assert abs(float(array.mean())) < 1.5
        assert array.min() < 0.0, "output is not centred; normalisation may be missing"

    @pytest.mark.parametrize("size", [64, 128, 224])
    def test_honours_configured_image_size(self, size: int) -> None:
        array = preprocess_image(make_image_bytes(), PreprocessConfig(image_size=size))
        assert array.shape == (3, size, size)


class TestStackBatch:
    def test_stacks_to_nchw(self) -> None:
        arrays = [preprocess_image(make_image_bytes(seed=i)) for i in range(3)]
        assert stack_batch(arrays).shape == (3, 3, 224, 224)

    def test_rejects_empty_batch(self) -> None:
        with pytest.raises(ValueError, match="empty batch"):
            stack_batch([])

    def test_rejects_ragged_batch(self) -> None:
        """Mismatched shapes must raise rather than be padded or truncated.

        A ragged batch means a preprocessing bug; silently reshaping would hide
        it and produce wrong predictions.
        """
        with pytest.raises(ValueError):
            stack_batch([np.zeros((3, 224, 224)), np.zeros((3, 128, 128))])


class TestSoftmax:
    def test_produces_a_probability_distribution(self) -> None:
        probabilities = softmax(np.array([1.0, 2.0, 3.0]))
        assert probabilities.sum() == pytest.approx(1.0)
        assert (probabilities > 0).all()

    def test_is_numerically_stable_for_large_logits(self) -> None:
        """Naive exp() overflows to inf here and yields nan.

        Large logits occur routinely in a confident, well-trained model, so
        this is a realistic input rather than a contrived one.
        """
        probabilities = softmax(np.array([1000.0, 1001.0, 999.0]))
        assert np.isfinite(probabilities).all()
        assert probabilities.sum() == pytest.approx(1.0)

    def test_matches_torch_reference(self) -> None:
        logits = np.random.default_rng(0).standard_normal(50).astype(np.float32)
        expected = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        assert np.allclose(softmax(logits), expected, atol=1e-6)

    def test_handles_batched_input(self) -> None:
        probabilities = softmax(np.random.default_rng(0).standard_normal((4, 10)))
        assert np.allclose(probabilities.sum(axis=-1), 1.0)


class TestTopK:
    def test_returns_descending_scores_and_indices(self) -> None:
        probabilities = np.array([0.1, 0.5, 0.2, 0.15, 0.05])
        scores, indices = top_k(probabilities, 3)

        assert list(indices) == [1, 2, 3]
        assert list(scores) == pytest.approx([0.5, 0.2, 0.15])
        assert (np.diff(scores) <= 0).all()

    def test_clamps_k_to_available_classes(self) -> None:
        scores, indices = top_k(np.array([0.6, 0.4]), 10)
        assert len(scores) == len(indices) == 2

    def test_matches_torch_topk(self) -> None:
        probabilities = softmax(np.random.default_rng(0).standard_normal(200))
        scores, indices = top_k(probabilities, 5)
        expected = torch.topk(torch.from_numpy(probabilities), 5)

        assert list(indices) == expected.indices.tolist()
        assert np.allclose(scores, expected.values.numpy())
