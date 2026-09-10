#!/usr/bin/env python3
"""Download and prepare the datasets used by this project.

This is a corrected and hardened replacement for the ``download_datasets.py``
shipped with the challenge repository. See ``README.md`` for a full write-up;
in summary, the original had two defects:

1. ``DatasetDownloader({})`` in the argparse block passed a ``dict`` to
   ``Path()``, raising ``TypeError`` before the parser was even built. The
   script could not run at all. The dataset catalogue does not depend on
   instance state, so it is a module-level constant here.

2. Tiny-ImageNet's validation split is not in ``ImageFolder`` layout — it is a
   flat ``val/images/`` directory plus a ``val_annotations.txt`` mapping. Any
   loader pointing ``ImageFolder`` at it silently finds one class and labels
   every image ``0``. :func:`restructure_tiny_imagenet_val` reorganises the
   split into per-class directories so validation metrics are meaningful.

Beyond the fixes, this version adds resumable downloads, atomic extraction,
path-traversal-safe archive handling, and honours the COCO ``sample_size``
that the original declared but ignored.

Usage
-----
    python scripts/setup/download_datasets.py --dataset tiny_imagenet
    python scripts/setup/download_datasets.py --dataset all
    python scripts/setup/download_datasets.py --dataset tiny_imagenet --data-dir /custom/path
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger("download_datasets")

CHUNK_SIZE = 1024 * 1024  # 1 MiB
DOWNLOAD_TIMEOUT = (10, 60)  # (connect, read) seconds


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a downloadable dataset."""

    name: str
    url: str
    filename: str
    extract_dir: str
    md5: str | None = None
    #: For COCO, cap the number of images retained after extraction.
    sample_size: int | None = None
    #: Extra archives fetched into the same directory (e.g. COCO annotations).
    extra_urls: tuple[tuple[str, str], ...] = field(default=())


# Module-level catalogue. Lifting this out of ``__init__`` is what fixes the
# original script's crash: the argparse ``choices`` list no longer requires
# constructing a throwaway downloader.
DATASETS: dict[str, DatasetSpec] = {
    "tiny_imagenet": DatasetSpec(
        name="tiny_imagenet",
        url="https://cs231n.stanford.edu/tiny-imagenet-200.zip",
        filename="tiny-imagenet-200.zip",
        extract_dir="tiny-imagenet-200",
        # Published checksum for the Stanford CS231n distribution.
        md5="90528d7ca1a48142e341f4ef8d21d0de",
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        filename="cifar-100-python.tar.gz",
        extract_dir="cifar-100-python",
        md5="eb9058c3a382ffc7106e4002c42a8d85",
    ),
    "coco_sample": DatasetSpec(
        name="coco_sample",
        url="http://images.cocodataset.org/zips/val2017.zip",
        filename="val2017.zip",
        extract_dir="coco_val2017",
        sample_size=1000,
        extra_urls=(
            (
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "annotations_trainval2017.zip",
            ),
        ),
    ),
}


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be downloaded or prepared."""


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def calculate_md5(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the hex MD5 digest of ``path``, read incrementally."""
    digest = hashlib.md5()  # noqa: S324 - integrity check, not a security control
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, resume: bool = True) -> None:
    """Stream ``url`` to ``destination`` with a progress bar.

    Downloads to a ``.part`` sidecar and renames on success, so an interrupted
    run never leaves a truncated file that looks complete. If a partial file
    exists and the server supports range requests, the transfer resumes.
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if resume and partial.exists() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = requests.get(url, stream=True, headers=headers, timeout=DOWNLOAD_TIMEOUT)

    # A server that ignores the Range header replies 200; restart from zero.
    if existing and response.status_code == 200:
        logger.debug("Server ignored range request; restarting download")
        existing = 0
    elif existing and response.status_code == 206:
        logger.info("Resuming from %.1f MiB", existing / 1024 / 1024)

    response.raise_for_status()

    remaining = int(response.headers.get("content-length", 0))
    total = remaining + existing if remaining else None

    mode = "ab" if existing else "wb"
    with (
        partial.open(mode) as handle,
        tqdm(
            desc=destination.name,
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress,
    ):
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                handle.write(chunk)
                progress.update(len(chunk))

    partial.rename(destination)


def verify(path: Path, expected_md5: str | None) -> None:
    """Raise :class:`DatasetError` if ``path`` does not match ``expected_md5``."""
    if expected_md5 is None:
        logger.debug("No checksum published for %s; skipping verification", path.name)
        return

    logger.info("Verifying checksum for %s", path.name)
    actual = calculate_md5(path)
    if actual != expected_md5:
        raise DatasetError(
            f"Checksum mismatch for {path.name}: expected {expected_md5}, got {actual}. "
            "The download may be corrupt or the upstream file may have changed."
        )


def _is_within(base: Path, target: Path) -> bool:
    """Return whether ``target`` resolves to a location inside ``base``."""
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def extract(archive: Path, destination: Path) -> None:
    """Extract ``archive`` into ``destination``, rejecting path traversal.

    Both ``zipfile`` and ``tarfile`` will happily write outside the target
    directory if an archive contains ``../`` entries or absolute paths (the
    "zip slip" class of bug). Every member is checked before extraction.
    """
    destination.mkdir(parents=True, exist_ok=True)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not _is_within(destination, destination / member):
                    raise DatasetError(f"Unsafe path in {archive.name}: {member!r}")
            # Safe: every member was checked against the destination root above.
            zf.extractall(destination)  # noqa: S202
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            # filter="data" (Python 3.12+) strips absolute paths, parent
            # traversal, symlinks, and device nodes.
            tf.extractall(destination, filter="data")
    else:
        raise DatasetError(f"Unsupported archive format: {archive.name}")


# ---------------------------------------------------------------------------
# Dataset-specific preparation
# ---------------------------------------------------------------------------
def flatten_single_root(staging: Path) -> None:
    """Collapse a redundant top-level directory produced by extraction.

    Most archives here wrap their contents in a directory named after the
    dataset, so extracting ``tiny-imagenet-200.zip`` into ``tiny-imagenet-200/``
    yields ``tiny-imagenet-200/tiny-imagenet-200/...``. When the staging
    directory contains exactly one entry and that entry is a directory, its
    children are lifted up one level.

    No-op when the archive has multiple top-level entries (e.g. COCO, where
    ``val2017/`` and ``annotations/`` are siblings).
    """
    entries = list(staging.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return

    nested = entries[0]
    logger.debug("Collapsing redundant top-level directory %s", nested.name)

    # Move to a temporary name first: on case-insensitive filesystems the
    # nested directory may collide with its parent during the rename.
    holding = staging.parent / f".{staging.name}-flatten"
    nested.rename(holding)
    staging.rmdir()
    holding.rename(staging)


def restructure_tiny_imagenet_val(root: Path) -> int:
    """Convert Tiny-ImageNet's validation split into ``ImageFolder`` layout.

    The distributed layout is::

        val/
          images/            <- all 10,000 images, flat
          val_annotations.txt  <- "<filename>\\t<wnid>\\t<bbox coords...>"

    ``torchvision.datasets.ImageFolder`` infers classes from subdirectory
    names, so pointing it at ``val/`` yields a single class named ``images``
    with every label set to ``0``. Validation accuracy computed against that is
    meaningless, and nothing raises — which is exactly what makes the bug
    dangerous. This rewrites the split to::

        val/
          n01443537/<images>
          n01629819/<images>
          ...

    Returns the number of images moved. Idempotent: a second call is a no-op.
    """
    val_dir = root / "val"
    images_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"

    if not val_dir.is_dir():
        raise DatasetError(f"Expected validation directory at {val_dir}")

    # Already restructured: `images/` is gone and class dirs exist.
    if not images_dir.is_dir():
        class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
        if class_dirs:
            logger.info("Validation split already restructured (%d classes)", len(class_dirs))
            return 0
        raise DatasetError(f"Validation split at {val_dir} is empty or malformed")

    if not annotations.is_file():
        raise DatasetError(f"Missing annotation file: {annotations}")

    logger.info("Restructuring Tiny-ImageNet validation split into ImageFolder layout")

    moved = 0
    with annotations.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.split("\t")
            if len(parts) < 2:
                logger.warning("Skipping malformed annotation at line %d", line_no)
                continue

            filename, wnid = parts[0], parts[1]
            source = images_dir / filename
            if not source.is_file():
                logger.warning("Annotated image not found: %s", filename)
                continue

            class_dir = val_dir / wnid
            class_dir.mkdir(exist_ok=True)
            source.rename(class_dir / filename)
            moved += 1

    # Remove the now-empty flat directory so ImageFolder does not see it as a
    # class. Anything left behind is unannotated and would corrupt labels.
    leftovers = list(images_dir.iterdir())
    if leftovers:
        logger.warning("%d unannotated images remain; discarding", len(leftovers))
    shutil.rmtree(images_dir)

    n_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
    logger.info("Restructured %d validation images into %d class folders", moved, n_classes)
    return moved


def subsample_coco(root: Path, sample_size: int) -> int:
    """Trim the extracted COCO image directory to ``sample_size`` images.

    The original script declared ``sample_size: 1000`` but never applied it,
    downloading and keeping all 5,000 val2017 images. We still download the
    full archive — COCO does not expose per-image range requests — but retain
    a deterministic subset (sorted by filename) so the working set is small and
    reproducible across machines.

    Returns the number of images retained.
    """
    image_dirs = [p for p in root.rglob("val2017") if p.is_dir()]
    if not image_dirs:
        logger.warning("No val2017 directory found under %s; skipping subsample", root)
        return 0

    images_dir = image_dirs[0]
    images = sorted(images_dir.glob("*.jpg"))
    if len(images) <= sample_size:
        logger.info("COCO already at %d images (<= %d)", len(images), sample_size)
        return len(images)

    logger.info("Subsampling COCO from %d to %d images", len(images), sample_size)
    for image in images[sample_size:]:
        image.unlink()

    return sample_size


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
class DatasetDownloader:
    """Downloads, verifies, extracts, and prepares datasets under ``data_dir``."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        #: Exposed for backwards compatibility with the original interface.
        self.datasets = DATASETS

    def download_dataset(self, name: str, *, force: bool = False) -> Path:
        """Fetch and prepare a single dataset. Returns its directory."""
        try:
            spec = DATASETS[name]
        except KeyError:
            known = ", ".join(sorted(DATASETS))
            raise DatasetError(f"Unknown dataset {name!r}. Available: {known}") from None

        target = self.data_dir / spec.extract_dir

        if target.exists() and not force:
            logger.info("%s already present at %s (use --force to re-download)", name, target)
            return target

        if force and target.exists():
            shutil.rmtree(target)

        # Extract into a temporary sibling and move into place only on success,
        # so a failure never leaves a half-populated dataset directory that the
        # "already present" check above would later trust.
        with tempfile.TemporaryDirectory(dir=self.data_dir, prefix=f".{name}-") as tmp:
            staging = Path(tmp) / spec.extract_dir

            archive = self.data_dir / spec.filename
            logger.info("Downloading %s", name)
            download(spec.url, archive)
            verify(archive, spec.md5)

            logger.info("Extracting %s", name)
            extract(archive, staging)
            archive.unlink()

            for extra_url, extra_name in spec.extra_urls:
                extra_archive = self.data_dir / extra_name
                logger.info("Downloading %s for %s", extra_name, name)
                download(extra_url, extra_archive)
                extract(extra_archive, staging)
                extra_archive.unlink()

            self._prepare(spec, staging)
            shutil.move(str(staging), str(target))

        logger.info("Dataset %s ready at %s", name, target)
        return target

    def _prepare(self, spec: DatasetSpec, root: Path) -> None:
        """Run dataset-specific post-extraction fixes."""
        flatten_single_root(root)

        if spec.name == "tiny_imagenet":
            restructure_tiny_imagenet_val(root)
        elif spec.name == "coco_sample" and spec.sample_size:
            subsample_coco(root, spec.sample_size)

    def download_all(self, *, force: bool = False) -> dict[str, Path | None]:
        """Fetch every dataset, continuing past individual failures."""
        results: dict[str, Path | None] = {}
        for name in DATASETS:
            try:
                results[name] = self.download_dataset(name, force=force)
            except (DatasetError, requests.RequestException, OSError) as exc:
                logger.error("Failed to download %s: %s", name, exc)
                results[name] = None
        return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download datasets for the ML Engineer challenge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=[*sorted(DATASETS), "all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    parser.add_argument("--data-dir", default="data", help="Target directory (default: data)")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    downloader = DatasetDownloader(args.data_dir)

    try:
        if args.dataset == "all":
            results = downloader.download_all(force=args.force)
            failed = [name for name, path in results.items() if path is None]
            if failed:
                logger.error("Failed datasets: %s", ", ".join(failed))
                return 1
        else:
            downloader.download_dataset(args.dataset, force=args.force)
    except KeyboardInterrupt:
        logger.warning("Interrupted; partial downloads are resumable on re-run")
        return 130
    except (DatasetError, requests.RequestException, OSError) as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
