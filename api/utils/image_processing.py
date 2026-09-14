"""Image preprocessing for inference.

Preprocessing must reproduce the *evaluation* transform used during training
exactly. Any divergence — a different resize filter, a different crop ratio,
the wrong normalisation statistics — degrades accuracy silently, with no error
and no obvious symptom beyond predictions that are worse than the benchmark
claimed. That makes it one of the highest-risk pieces of a serving stack.

The pipeline implemented here mirrors
:func:`models.data.augmentation.build_eval_transform`:

1. Convert to RGB.
2. Resize the short edge to ``image_size / 0.875``.
3. Centre-crop to ``image_size``.
4. Scale to ``[0, 1]``.
5. Normalise with the training statistics.

It is implemented with NumPy and Pillow rather than by importing the training
transform, because the serving image deliberately does not install torch or
torchvision. Duplicating a handful of well-understood operations is the price
of keeping several gigabytes of training dependencies out of the production
container; :func:`preprocess_image` is covered by a test that asserts numerical
agreement with the torchvision pipeline.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

#: Must match models.data.augmentation.IMAGENET_MEAN / _STD.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Standard resize-then-centre-crop ratio, matching the evaluation transform.
CROP_RATIO = 0.875


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Preprocessing parameters, normally loaded from model metadata."""

    image_size: int = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    crop_ratio: float = CROP_RATIO


def decode_image(data: bytes) -> Image.Image:
    """Decode bytes to an RGB image.

    Conversion to RGB is unconditional. Greyscale, palette, and RGBA inputs
    would otherwise produce arrays with the wrong channel count and fail deep
    inside the model with an unhelpful shape error. Alpha is discarded rather
    than composited: the models were trained on opaque images, and inventing a
    background colour would be a silent, arbitrary choice.
    """
    opened = Image.open(io.BytesIO(data))
    return opened.convert("RGB") if opened.mode != "RGB" else opened


def resize_short_edge(image: Image.Image, target: int) -> Image.Image:
    """Resize so the shorter edge equals ``target``, preserving aspect ratio.

    Uses **bilinear** resampling with antialiasing, which is what
    ``torchvision.transforms.v2.Resize`` does by default. This is worth stating
    explicitly because it is easy to assume bicubic: switching to bicubic here
    changes activations by up to 2.5 in normalised units on high-frequency
    inputs, which is a silent accuracy loss with no error and no obvious
    symptom. ``tests/unit/test_image_processing.py`` asserts agreement with the
    torchvision pipeline to guard against exactly that drift.

    Pillow's ``BILINEAR`` applies a proper support-scaled filter when
    downsampling, matching torchvision's ``antialias=True``.
    """
    width, height = image.size
    if width <= height:
        new_width = target
        new_height = max(1, round(height * target / width))
    else:
        new_height = target
        new_width = max(1, round(width * target / height))
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """Crop a centred square of ``size`` pixels.

    The offset is ``round((dimension - size) / 2)``, matching torchvision's
    ``CenterCrop`` exactly. This is not interchangeable with floor division:
    when ``dimension - size`` is odd the two disagree by one pixel — for a
    275-wide image, ``round(25.5) = 26`` against ``51 // 2 = 25``.

    A one-pixel horizontal shift produces no error and no visible defect, but
    it does shift every activation slightly away from the distribution the
    model was trained on. It is exactly the class of preprocessing bug that
    surfaces only as accuracy that is inexplicably below the benchmark.
    """
    width, height = image.size
    left = round((width - size) / 2)
    top = round((height - size) / 2)
    return image.crop((left, top, left + size, top + size))


def to_normalised_array(
    image: Image.Image,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    """Convert to a normalised CHW float32 array.

    Returns shape ``(3, H, W)``. Division by 255 happens before normalisation,
    matching ``ToDtype(scale=True)`` followed by ``Normalize``.
    """
    array = np.asarray(image, dtype=np.float32) / 255.0  # HWC in [0, 1]
    array = array.transpose(2, 0, 1)  # CHW
    mean_array = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return (array - mean_array) / std_array


def preprocess_image(data: bytes, config: PreprocessConfig | None = None) -> np.ndarray:
    """Decode and preprocess one image into a model-ready array.

    Returns shape ``(3, image_size, image_size)``, float32. Callers batch these
    with :func:`stack_batch`.
    """
    config = config or PreprocessConfig()
    image = decode_image(data)

    resize_target = int(np.floor(config.image_size / config.crop_ratio))
    image = resize_short_edge(image, resize_target)
    image = center_crop(image, config.image_size)

    return to_normalised_array(image, config.mean, config.std)


def stack_batch(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack preprocessed images into a ``(N, 3, H, W)`` batch.

    ``np.stack`` raises on mismatched shapes, which is the desired behaviour: a
    ragged batch means a preprocessing bug, and silently padding or truncating
    would hide it.
    """
    if not arrays:
        raise ValueError("Cannot stack an empty batch")
    return np.stack(arrays, axis=0)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Subtracting the row max before exponentiating prevents overflow to ``inf``
    for large logits; the result is mathematically identical.
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def top_k(probabilities: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` highest probabilities and their indices, descending.

    Uses ``argpartition`` for the selection (linear) and sorts only the ``k``
    selected entries, rather than sorting all classes.
    """
    k = min(k, probabilities.shape[-1])
    indices = np.argpartition(-probabilities, kth=k - 1, axis=-1)[..., :k]
    selected = np.take_along_axis(probabilities, indices, axis=-1)

    order = np.argsort(-selected, axis=-1)
    return (
        np.take_along_axis(selected, order, axis=-1),
        np.take_along_axis(indices, order, axis=-1),
    )
