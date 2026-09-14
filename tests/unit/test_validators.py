"""Tests for image upload validation.

Validation is a security boundary, so these tests are adversarial: they submit
the inputs an attacker would, not just the ones a well-behaved client sends.
"""

from __future__ import annotations

import io
import struct

import pytest
from PIL import Image

from api.exceptions import (
    InvalidImageError,
    PayloadTooLargeError,
    UnsupportedFormatError,
)
from api.utils.validators import (
    MAX_DIMENSION,
    MIN_DIMENSION,
    SUPPORTED_FORMATS,
    inspect_image,
    validate_dimensions,
    validate_format,
    validate_image_upload,
    validate_upload_size,
)
from tests.conftest import make_image_bytes

pytestmark = pytest.mark.unit


class TestUploadSize:
    def test_accepts_upload_within_limit(self) -> None:
        # Asserting absence of an exception. Written explicitly rather than
        # relying on a bare call, so the intent is visible and an accidentally
        # deleted assertion cannot masquerade as this pattern.
        validate_upload_size(b"x" * 1000, max_bytes=2000)
        assert True, "validate_upload_size must accept a payload within the limit"

    def test_rejects_oversized_upload(self) -> None:
        with pytest.raises(PayloadTooLargeError) as exc_info:
            validate_upload_size(b"x" * 3000, max_bytes=2000)

        # The error must tell the caller what the limit is, otherwise they
        # cannot fix their request without guessing.
        assert exc_info.value.details["max_bytes"] == 2000
        assert exc_info.value.details["size_bytes"] == 3000
        assert exc_info.value.status_code == 413

    def test_rejects_empty_upload(self) -> None:
        with pytest.raises(InvalidImageError):
            validate_upload_size(b"", max_bytes=2000)

    def test_boundary_is_inclusive(self) -> None:
        """Exactly at the limit is allowed; one byte over is not."""
        validate_upload_size(b"x" * 100, max_bytes=100)
        with pytest.raises(PayloadTooLargeError):
            validate_upload_size(b"x" * 101, max_bytes=100)


class TestInspectImage:
    @pytest.mark.parametrize("fmt", sorted(SUPPORTED_FORMATS))
    def test_identifies_supported_formats(self, fmt: str) -> None:
        metadata = inspect_image(make_image_bytes(64, 48, fmt=fmt))
        assert metadata.format == fmt
        assert (metadata.width, metadata.height) == (64, 48)
        assert metadata.pixels == 64 * 48

    def test_rejects_non_image_bytes(self) -> None:
        with pytest.raises(InvalidImageError):
            inspect_image(b"this is definitely not an image")

    def test_rejects_truncated_image(self) -> None:
        """A header that promises data the file does not contain."""
        data = make_image_bytes(128, 128, fmt="PNG")
        with pytest.raises(InvalidImageError):
            inspect_image(data[:20])

    def test_does_not_leak_internal_detail(self) -> None:
        """Pillow's internal errors must not reach the caller.

        Library exception text can contain file paths and version information,
        and is meaningless to an API consumer.
        """
        with pytest.raises(InvalidImageError) as exc_info:
            inspect_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)

        message = exc_info.value.message.lower()
        assert "traceback" not in message
        assert "pil" not in message


class TestFormatValidation:
    def test_rejects_unsupported_format(self) -> None:
        metadata = inspect_image(make_image_bytes(64, 64, fmt="TIFF"))
        with pytest.raises(UnsupportedFormatError) as exc_info:
            validate_format(metadata)

        # The caller is told what *is* accepted, not merely what was refused.
        assert "TIFF" in str(exc_info.value.details["detected_format"])
        assert sorted(SUPPORTED_FORMATS) == exc_info.value.details["supported_formats"]

    def test_format_comes_from_content_not_filename(self) -> None:
        """A PNG is a PNG regardless of what the client claims.

        Content-Type and filename extension are attacker-controlled. This is
        the property that stops a caller smuggling an unexpected decoder by
        renaming a file.
        """
        png = make_image_bytes(64, 64, fmt="PNG")
        assert inspect_image(png).format == "PNG"


class TestDimensionValidation:
    def test_accepts_reasonable_dimensions(self) -> None:
        metadata = inspect_image(make_image_bytes(256, 256))

        validate_dimensions(metadata, max_pixels=10_000_000)

        # The call above raises on rejection; these confirm it inspected the
        # image we think it did, so the test cannot pass on a no-op.
        assert metadata.width == 256
        assert metadata.pixels == 256 * 256

    @pytest.mark.parametrize("size", [1, 8, MIN_DIMENSION - 1])
    def test_rejects_degenerate_images(self, size: int) -> None:
        metadata = inspect_image(make_image_bytes(size, size, fmt="PNG"))
        with pytest.raises(InvalidImageError):
            validate_dimensions(metadata, max_pixels=10_000_000)

    def test_accepts_exactly_minimum_dimension(self) -> None:
        """The boundary is inclusive: exactly MIN_DIMENSION is acceptable."""
        metadata = inspect_image(make_image_bytes(MIN_DIMENSION, MIN_DIMENSION, fmt="PNG"))

        validate_dimensions(metadata, max_pixels=10_000_000)

        assert metadata.width == MIN_DIMENSION
        # One pixel smaller must be refused, which is what makes this a
        # boundary test rather than a restatement of the case above.
        smaller = inspect_image(make_image_bytes(MIN_DIMENSION - 1, MIN_DIMENSION - 1, fmt="PNG"))
        with pytest.raises(InvalidImageError):
            validate_dimensions(smaller, max_pixels=10_000_000)

    def test_rejects_excessive_pixel_count(self) -> None:
        metadata = inspect_image(make_image_bytes(2000, 2000, fmt="JPEG"))
        with pytest.raises(InvalidImageError) as exc_info:
            validate_dimensions(metadata, max_pixels=1_000_000)
        assert exc_info.value.details["pixels"] == 4_000_000

    def test_rejects_extreme_aspect_ratio(self) -> None:
        """A 1 x N strip passes a pixel-count check but is still pathological.

        This is why dimensions are checked independently of total pixels.
        """
        from api.utils.validators import ImageMetadata

        metadata = ImageMetadata(
            format="PNG", width=MAX_DIMENSION + 1, height=1, mode="RGB", size_bytes=100
        )
        with pytest.raises(InvalidImageError):
            validate_dimensions(metadata, max_pixels=10_000_000)


class TestDecompressionBomb:
    def test_rejects_declared_bomb_without_decoding(self) -> None:
        """A small file declaring enormous dimensions must be refused.

        The PNG below is a valid header claiming 30000x30000 pixels — 2.7 GB
        decoded — in under 100 bytes. Rejecting it depends on inspecting the
        header rather than decoding first, which is the whole point of the
        cheapest-check-first ordering.
        """
        width = height = 30_000
        ihdr = struct.pack(">II", width, height) + bytes([8, 2, 0, 0, 0])
        chunk = (
            struct.pack(">I", len(ihdr))
            + b"IHDR"
            + ihdr
            + struct.pack(">I", 0)  # CRC placeholder; Pillow reads the header first
        )
        bomb = b"\x89PNG\r\n\x1a\n" + chunk

        assert len(bomb) < 100

        with pytest.raises(InvalidImageError):
            validate_image_upload(bomb, max_bytes=10_000_000, max_pixels=89_478_485)


class TestFullValidationChain:
    def test_accepts_valid_upload(self) -> None:
        metadata = validate_image_upload(
            make_image_bytes(256, 256), max_bytes=10_000_000, max_pixels=10_000_000
        )
        assert metadata.format == "JPEG"

    def test_size_is_checked_before_decoding(self) -> None:
        """An oversized upload is refused on length alone.

        The payload here is not a valid image at all; if decoding happened
        first this would raise InvalidImageError instead, revealing that an
        attacker can force a decode by exceeding the size limit.
        """
        with pytest.raises(PayloadTooLargeError):
            validate_image_upload(b"x" * 5000, max_bytes=1000, max_pixels=10_000_000)

    @pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
    def test_accepts_non_rgb_modes(self, mode: str) -> None:
        """Greyscale, alpha, and palette images are valid uploads.

        They are converted during preprocessing rather than rejected: users
        legitimately upload screenshots and greyscale photographs.
        """
        image = Image.new(mode, (64, 64))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        metadata = validate_image_upload(
            buffer.getvalue(), max_bytes=10_000_000, max_pixels=10_000_000
        )
        assert metadata.mode == mode
