#!/usr/bin/env python3
"""Assemble serving artifacts from a training run.

The API loads models from a flat artifacts directory that is mounted into the
container. This script populates it from a training run:

    models/artifacts/
      labels.json                  class names and WordNet IDs
      metrics.json                 validation metrics, surfaced at /models
      onnx/classifier_fp32.onnx    the served graph
      onnx/classifier_int8.onnx    optional quantized variant

Separating "what was trained" from "what is served" matters: the serving
container never sees training checkpoints, optimizer state, or the dataset, and
the artifacts directory can be published to a registry or volume independently
of the run that produced it.

Usage::

    uv run python scripts/prepare_artifacts.py
    uv run python scripts/prepare_artifacts.py --run-dir models/artifacts/runs/<run>
    uv run python scripts/prepare_artifacts.py --skip-int8
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("prepare_artifacts")


def find_latest_run(runs_dir: Path) -> Path:
    """Return the most recently modified run that has a config.json."""
    candidates = [p for p in runs_dir.glob("*") if (p / "config.json").is_file()]
    if not candidates:
        raise SystemExit(
            f"No training runs found under {runs_dir}.\n"
            f"Train a model first:\n  uv run python -m models.training.train"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Training run to publish")
    parser.add_argument("--runs-dir", type=Path, default=Path("models/artifacts/runs"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument(
        "--version",
        default="v1",
        help=(
            "Model version to publish. `v1` uses the flat artefact layout the "
            "serving path already understands. Any other value writes "
            "`onnx/<version>/` with its own labels.json, which is what a live "
            "rollout actually ships."
        ),
    )
    parser.add_argument("--skip-int8", action="store_true", help="Do not produce the INT8 variant")
    parser.add_argument(
        "--with-detection",
        action="store_true",
        help="Also export the RT-DETR object detector (downloads ~80 MB)",
    )
    parser.add_argument(
        "--with-similarity",
        action="store_true",
        help="Also export the embedding model for similarity search",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Export every model: classifier, detector, and embedder",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=256,
        help="Images used to calibrate INT8 activation ranges",
    )
    parser.add_argument("--force", action="store_true", help="Re-export if present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    run_dir = args.run_dir or find_latest_run(args.runs_dir)
    artifacts_dir = args.artifacts_dir
    version = args.version
    # v1 keeps the flat layout the serving path already understands. Any other
    # version is self-describing: onnx/<version>/ carries its own graph and
    # labels.json, which is the contract `resolve_versioned_artifact` enforces.
    if version == "v1":
        onnx_dir = artifacts_dir / "onnx"
        labels_dir = artifacts_dir
    else:
        onnx_dir = artifacts_dir / "onnx" / version
        labels_dir = onnx_dir
    onnx_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Publishing artifacts from %s as %s", run_dir, version)

    # --- Labels -----------------------------------------------------------
    # The serving container has no dataset, so the index-to-name mapping must
    # travel with the model. Re-deriving it from directory listings at serving
    # time would silently permute every label if the two ever differed.
    config = json.loads((run_dir / "config.json").read_text())
    task = config.get("task", "classification")
    labels_src = run_dir / "labels.json"
    # Classification owns the shared labels.json / metrics.json at the
    # artefacts root. An embedding or detection run writing them would
    # overwrite the classifier's class list with a different mapping.
    publish_shared_metadata = task == "classification" or version != "v1"
    if labels_src.is_file() and publish_shared_metadata:
        labels = json.loads(labels_src.read_text())
        labels["image_size"] = config["dataloader"]["image_size"]
        labels["model_name"] = config["model"]["name"]
        labels["version"] = version
        (labels_dir / "labels.json").write_text(json.dumps(labels, indent=2))
        logger.info(
            "Wrote labels.json (%d classes, image_size=%d) for %s",
            len(labels.get("class_names", [])),
            labels["image_size"],
            version,
        )
    else:
        labels = {"image_size": config["dataloader"]["image_size"]}
        if labels_src.is_file() and not publish_shared_metadata:
            logger.info("Leaving shared labels.json untouched (task=%s)", task)

    metrics_src = run_dir / "metrics.json"
    if metrics_src.is_file() and publish_shared_metadata:
        shutil.copy2(metrics_src, labels_dir / "metrics.json")
        logger.info("Copied metrics.json")

    # --- ONNX export ------------------------------------------------------
    from models.optimisation.export import export_onnx, load_classifier_from_run

    fp32_path = onnx_dir / "classifier_fp32.onnx"
    if task != "classification":
        logger.info("Skipping classifier export for task=%s", task)
    elif fp32_path.exists() and not args.force:
        logger.info("%s already exists (use --force to re-export)", fp32_path)
    else:
        model, metadata = load_classifier_from_run(run_dir)
        max_diff = export_onnx(model, fp32_path, image_size=metadata["image_size"])
        logger.info("Exported FP32 ONNX, max |torch - onnx| = %.3e", max_diff)

    # --- INT8 -------------------------------------------------------------
    int8_path = onnx_dir / "classifier_int8.onnx"
    if task != "classification" or args.skip_int8:
        logger.info("Skipping classifier INT8 export")
    elif int8_path.exists() and not args.force:
        logger.info("%s already exists (use --force to re-export)", int8_path)
    else:
        _export_int8(fp32_path, int8_path, labels["image_size"], args.calibration_samples)

    # --- Detection --------------------------------------------------------
    detector_path = onnx_dir / "detector_fp32.onnx"
    want_detector = args.with_detection or args.all_models
    if want_detector and (not detector_path.exists() or args.force):
        from models.optimisation.export_detection import (
            export_detection_onnx,
            write_detection_metadata,
        )

        detector_run = run_dir if task == "detection" else None
        metadata = export_detection_onnx(detector_path, run_dir=detector_run)
        write_detection_metadata(labels_dir, metadata)
        logger.info(
            "Exported detector: %d classes, max |torch - onnx| = %.3e",
            metadata["num_classes"],
            metadata["max_abs_diff"],
        )
    elif detector_path.exists():
        logger.info("%s already exists", detector_path)

    detector_int8_path = onnx_dir / "detector_int8.onnx"
    if not want_detector:
        pass
    elif args.skip_int8:
        logger.info("Skipping detector INT8 export")
    elif detector_int8_path.exists() and not args.force:
        logger.info("%s already exists (use --force to re-export)", detector_int8_path)
    elif not detector_path.exists():
        logger.warning("Skipping detector INT8: %s is missing", detector_path)
    else:
        _export_detector_int8(detector_path, detector_int8_path, args.calibration_samples)

    # --- Similarity embeddings ---------------------------------------------
    embedder_path = onnx_dir / "embedder_fp32.onnx"
    want_embedder = args.with_similarity or args.all_models
    if want_embedder and (not embedder_path.exists() or args.force):
        from models.optimisation.export_embedding import export_embedding_onnx

        embedding_meta = export_embedding_onnx(run_dir, embedder_path)
        logger.info(
            "Exported embedder: %d-d features, max |torch - onnx| = %.3e",
            embedding_meta["embedding_dim"],
            embedding_meta["max_abs_diff"],
        )
        if not (artifacts_dir / "similarity.index").exists():
            logger.info(
                "No similarity index yet. Build one with:\n"
                "  uv run python scripts/build_similarity_index.py"
            )
    elif embedder_path.exists():
        logger.info("%s already exists", embedder_path)

    embedder_int8_path = onnx_dir / "embedder_int8.onnx"
    if not want_embedder:
        pass
    elif args.skip_int8:
        logger.info("Skipping embedder INT8 export")
    elif embedder_int8_path.exists() and not args.force:
        logger.info("%s already exists (use --force to re-export)", embedder_int8_path)
    elif not embedder_path.exists():
        logger.warning("Skipping embedder INT8: %s is missing", embedder_path)
    else:
        _export_int8(
            embedder_path,
            embedder_int8_path,
            labels["image_size"],
            args.calibration_samples,
            task="embedding",
        )

    logger.info("Artifacts ready in %s", artifacts_dir)
    for path in sorted(artifacts_dir.rglob("*")):
        if path.is_file() and "runs" not in path.parts:
            logger.info(
                "  %-40s %6.1f MB", path.relative_to(artifacts_dir), path.stat().st_size / 1e6
            )

    return 0


def _export_int8(
    fp32_path: Path,
    int8_path: Path,
    image_size: int,
    calibration_samples: int,
    *,
    task: str = "classification",
) -> None:
    """Quantize a Tiny-ImageNet graph to INT8, warning about the measured cost.

    Requires the dataset for calibration. If it is unavailable the step is
    skipped rather than failing: FP32 is the default serving backend, so a
    missing INT8 variant degrades nothing.
    """
    try:
        from models.data.augmentation import AugmentationConfig, build_eval_transform
        from models.data.tiny_imagenet import build_datasets
        from models.optimisation.quantize import (
            TensorDatasetCalibrationReader,
            quantize_int8,
        )

        cfg = AugmentationConfig(image_size=image_size)
        data = build_datasets("data", cfg)
        calibration_set = data.train
        calibration_set.transform = build_eval_transform(cfg)

        reader = TensorDatasetCalibrationReader(
            calibration_set, input_name="images", num_samples=calibration_samples
        )
        quantize_int8(fp32_path, int8_path, reader)

        if task == "embedding":
            logger.warning(
                "INT8 embedder written, but recall@5 against the FP32 index drops "
                "by roughly half. It is opt-in via INFERENCE_BACKEND=onnx-int8; "
                "see benchmarks/ANALYSIS.md."
            )
        else:
            logger.warning(
                "INT8 artifact written, but it costs roughly 5.4 points of top-1 accuracy "
                "on this model and is slower than every GPU backend. It is not the "
                "recommended serving backend; see benchmarks/ANALYSIS.md."
            )
    except Exception as exc:
        logger.warning(
            "Skipping INT8 export (%s: %s). FP32 remains available and is the default.",
            type(exc).__name__,
            exc,
        )


def _export_detector_int8(fp32_path: Path, int8_path: Path, calibration_samples: int) -> None:
    """Quantize the detector, calibrating through the serving preprocess.

    RT-DETR skips mean/std normalisation and resizes without preserving aspect
    ratio, so calibration must use ``preprocess_for_detection``. Sample count is
    capped: percentile histograms over 640x640 activations exhaust host memory
    at the classifier's default of 256.
    """
    try:
        from api.utils.image_processing import preprocess_for_detection
        from models.optimisation.quantize import ImageFileCalibrationReader, quantize_int8

        image_dir = Path("data/coco_val2017/val2017")
        images = sorted(image_dir.glob("*.jpg"))
        if not images:
            raise FileNotFoundError(
                f"No COCO images under {image_dir}. Download them with:\n"
                "  python scripts/setup/download_datasets.py --dataset coco_sample"
            )

        reader = ImageFileCalibrationReader(
            images,
            input_name="pixel_values",
            preprocess=lambda data: preprocess_for_detection(data)[0],
            num_samples=min(calibration_samples, 64),
        )
        quantize_int8(fp32_path, int8_path, reader)
        logger.warning(
            "INT8 detector written, but COCO mAP collapses (~87% relative). "
            "It is opt-in via INFERENCE_BACKEND=onnx-int8; see benchmarks/ANALYSIS.md."
        )
    except Exception as exc:
        logger.warning(
            "Skipping detector INT8 export (%s: %s). FP32 remains available and is the default.",
            type(exc).__name__,
            exc,
        )


if __name__ == "__main__":
    sys.exit(main())
