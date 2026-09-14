#!/usr/bin/env python3
"""Build the image similarity index.

Embeds a subset of the Tiny-ImageNet training split with the fine-tuned
backbone and writes a FAISS index plus the label/path manifest the API needs to
turn a row number back into a meaningful result.

The *training* split is indexed, not validation. Indexing validation images
would make any evaluation of retrieval quality circular: a query drawn from the
validation set would retrieve itself.

Usage::

    uv run python scripts/build_similarity_index.py
    uv run python scripts/build_similarity_index.py --num-images 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("build_similarity_index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Training run supplying the backbone")
    parser.add_argument("--runs-dir", type=Path, default=Path("models/artifacts/runs"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--num-images",
        type=int,
        default=20_000,
        help="Images to index. Sampled evenly across classes.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--append",
        type=Path,
        default=None,
        help=(
            "Directory of images to add to the existing index instead of "
            "rebuilding. Costs only the new images' embedding time."
        ),
    )
    parser.add_argument(
        "--append-label",
        type=int,
        default=-1,
        help="Class index for appended images; -1 marks them as unlabelled",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    from scripts.prepare_artifacts import find_latest_run

    run_dir = args.run_dir or find_latest_run(args.runs_dir)
    logger.info("Using backbone from %s", run_dir)

    import onnxruntime as ort

    from api.utils.image_processing import PreprocessConfig, preprocess_image, stack_batch
    from models.data.tiny_imagenet import resolve_root
    from models.optimisation.export_embedding import build_index

    embedder_path = args.artifacts_dir / "onnx" / "embedder_fp32.onnx"
    if not embedder_path.exists():
        raise SystemExit(
            f"No embedding model at {embedder_path}. Run:\n"
            f"  uv run python scripts/prepare_artifacts.py --with-similarity"
        )

    session = ort.InferenceSession(str(embedder_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if args.append is not None:
        return _append_images(session, input_name, args)

    root = resolve_root(args.data_dir)
    train_dir = root / "train"

    # Sample evenly across classes rather than taking the first N files. An
    # alphabetical prefix would index only the first few dozen categories and
    # make retrieval useless for everything else.
    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    per_class = max(1, args.num_images // len(class_dirs))
    logger.info("Sampling %d images from each of %d classes", per_class, len(class_dirs))

    files: list[Path] = []
    labels: list[int] = []
    for class_index, class_dir in enumerate(class_dirs):
        images = sorted(class_dir.rglob("*.JPEG"))[:per_class]
        files.extend(images)
        labels.extend([class_index] * len(images))

    logger.info("Embedding %d images", len(files))

    config = PreprocessConfig(image_size=224)
    embeddings = np.empty((len(files), 384), dtype=np.float32)

    for start in range(0, len(files), args.batch_size):
        chunk = files[start : start + args.batch_size]
        batch = stack_batch([preprocess_image(path.read_bytes(), config) for path in chunk])
        embeddings[start : start + len(chunk)] = session.run(None, {input_name: batch})[0]

        if start % (args.batch_size * 50) == 0:
            logger.info("  %d / %d", start, len(files))

    metadata = build_index(
        embeddings,
        labels,
        [str(path.relative_to(root)) for path in files],
        args.artifacts_dir,
        source_model="tiny-imagenet-classifier:v1",
        image_size=224,
    )

    logger.info("Index ready: %d vectors", metadata.num_vectors)
    return 0


def _append_images(session: Any, input_name: str, args: argparse.Namespace) -> int:
    """Embed a directory of images and append them to the existing index."""
    from api.utils.image_processing import PreprocessConfig, preprocess_image, stack_batch
    from models.optimisation.export_embedding import append_to_index

    files = sorted(
        p for p in args.append.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise SystemExit(f"No images found under {args.append}")

    logger.info("Appending %d images from %s", len(files), args.append)

    config = PreprocessConfig(image_size=224)
    embeddings = np.empty((len(files), 384), dtype=np.float32)

    for start in range(0, len(files), args.batch_size):
        chunk = files[start : start + args.batch_size]
        batch = stack_batch([preprocess_image(p.read_bytes(), config) for p in chunk])
        embeddings[start : start + len(chunk)] = session.run(None, {input_name: batch})[0]

    metadata = append_to_index(
        args.artifacts_dir,
        embeddings,
        labels=[args.append_label] * len(files),
        paths=[str(p) for p in files],
    )
    logger.info("Index now holds %d vectors", metadata.num_vectors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
