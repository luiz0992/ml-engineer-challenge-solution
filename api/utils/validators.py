"""Image upload validation.

Validating untrusted image uploads is a security boundary, not a formality.
Three distinct risks are handled here:

* **Decompression bombs.** A few-kilobyte PNG can declare dimensions that
  expand to tens of gigabytes when decoded, exhausting memory and taking the
  worker down. Pillow's ``MAX_IMAGE_PIXELS`` guard is enabled, and dimensions
  are checked from the header *before* any pixel data is decoded.
* **Format confusion.** The client-supplied ``Content-Type`` and filename
  extension are attacker-controlled and are never trusted. Format is determined
  from the decoded image's own magic bytes.
* **Resource exhaustion.** Size is checked against a configured limit before the
  bytes are decoded.

Validation is deliberately ordered cheapest-first: byte length, then header
inspection, then full decode. An oversized or malformed upload is rejected
before the expensive step.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import PIL
from PIL import Image, UnidentifiedImageError

from api.exceptions import (
    InvalidImageError,
    PayloadTooLargeError,
    UnsupportedFormatError,
)

#: Formats accepted for upload. Deliberately narrow: each additional decoder is
#: additional attack surface, and these cover essentially all real traffic.
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP"})

#: Smallest edge accepted. Anything smaller carries no usable signal once
#: resized to the model's input resolution and is almost certainly a mistake.
MIN_DIMENSION = 16

#: Largest edge accepted before resizing. Independent of the total-pixel guard:
#: a 1 x 500,000 image passes a pixel-count check but is still pathological.
MAX_DIMENSION = 10_000


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Facts about a validated upload, derived from the image itself."""

    format: str
    width: int
    height: int
    mode: str
    size_bytes: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


def configure_pillow_limits(max_pixels: int) -> None:
    """Set Pillow's global decompression-bomb threshold.

    Above this, Pillow raises ``DecompressionBombError`` during decode. It is a
    process-global setting, so it is configured once at application startup
    rather than per request.
    """
    Image.MAX_IMAGE_PIXELS = max_pixels


def validate_upload_size(data: bytes, max_bytes: int) -> None:
    """Reject uploads larger than ``max_bytes``.

    Checked before decoding, so a hostile upload costs only the bytes already
    received.
    """
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"Upload is {len(data) / 1024 / 1024:.1f} MiB, which exceeds the "
            f"{max_bytes / 1024 / 1024:.1f} MiB limit.",
            details={"size_bytes": len(data), "max_bytes": max_bytes},
        )
    if not data:
        raise InvalidImageError("The uploaded file is empty.")


def inspect_image(data: bytes) -> ImageMetadata:
    """Read image metadata without decoding pixel data.

    ``Image.open`` is lazy: it parses the header and returns immediately. That
    lets dimensions and format be checked before committing to a decode that
    could allocate gigabytes.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            mode = image.mode
    except UnidentifiedImageError as exc:
        raise InvalidImageError("The uploaded file could not be identified as an image.") from exc
    except PIL.Image.DecompressionBombError as exc:
        raise InvalidImageError(
            "The image declares dimensions large enough to be a decompression bomb."
        ) from exc
    except Exception as exc:
        # Pillow raises a wide and version-dependent range of errors on
        # malformed input; none of them should reach the client verbatim.
        raise InvalidImageError("The uploaded file could not be read as an image.") from exc

    return ImageMetadata(
        format=image_format,
        width=width,
        height=height,
        mode=mode,
        size_bytes=len(data),
    )


def validate_format(metadata: ImageMetadata) -> None:
    """Reject formats outside the supported set.

    Uses the format Pillow detected from the file's own bytes, never the
    client-supplied Content-Type or filename extension.
    """
    if metadata.format not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Format {metadata.format or 'unknown'} is not supported.",
            details={
                "detected_format": metadata.format or None,
                "supported_formats": sorted(SUPPORTED_FORMATS),
            },
        )


def validate_dimensions(metadata: ImageMetadata, *, max_pixels: int) -> None:
    """Reject images that are degenerate or pathologically large."""
    if metadata.width < MIN_DIMENSION or metadata.height < MIN_DIMENSION:
        raise InvalidImageError(
            f"Image is {metadata.width}x{metadata.height}; the minimum is "
            f"{MIN_DIMENSION}x{MIN_DIMENSION}.",
            details={"width": metadata.width, "height": metadata.height},
        )

    if metadata.width > MAX_DIMENSION or metadata.height > MAX_DIMENSION:
        raise InvalidImageError(
            f"Image is {metadata.width}x{metadata.height}; the maximum edge length "
            f"is {MAX_DIMENSION}.",
            details={"width": metadata.width, "height": metadata.height},
        )

    if metadata.pixels > max_pixels:
        raise InvalidImageError(
            f"Image has {metadata.pixels:,} pixels, exceeding the {max_pixels:,} pixel limit.",
            details={"pixels": metadata.pixels, "max_pixels": max_pixels},
        )


def validate_image_upload(data: bytes, *, max_bytes: int, max_pixels: int) -> ImageMetadata:
    """Run the full validation chain, cheapest check first.

    Returns the validated metadata. Raises an :class:`~api.exceptions.APIError`
    subclass describing the first failure.
    """
    validate_upload_size(data, max_bytes)
    metadata = inspect_image(data)
    validate_format(metadata)
    validate_dimensions(metadata, max_pixels=max_pixels)
    return metadata
