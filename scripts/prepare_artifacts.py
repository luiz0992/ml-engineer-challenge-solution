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
    parser.add_argument("--skip-int8", action="store_true", help="Do not produce the INT8 variant")
    parser.add_argument(
        "--with-detection",
        action="store_true",
        help="Also export the RT-DETR object detector (downloads ~80 MB)",
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
    onnx_dir = artifacts_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Publishing artifacts from %s", run_dir)

    # --- Labels -----------------------------------------------------------
    # The serving container has no dataset, so the index-to-name mapping must
    # travel with the model. Re-deriving it from directory listings at serving
    # time would silently permute every label if the two ever differed.
    labels = json.loads((run_dir / "labels.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    labels["image_size"] = config["dataloader"]["image_size"]
    labels["model_name"] = config["model"]["name"]

    (artifacts_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    logger.info(
        "Wrote labels.json (%d classes, image_size=%d)",
        len(labels["class_names"]),
        labels["image_size"],
    )

    metrics_src = run_dir / "metrics.json"
    if metrics_src.is_file():
        shutil.copy2(metrics_src, artifacts_dir / "metrics.json")
        logger.info("Copied metrics.json")

    # --- ONNX export ------------------------------------------------------
    from models.optimisation.export import export_onnx, load_classifier_from_run

    fp32_path = onnx_dir / "classifier_fp32.onnx"
    if fp32_path.exists() and not args.force:
        logger.info("%s already exists (use --force to re-export)", fp32_path)
    else:
        model, metadata = load_classifier_from_run(run_dir)
        max_diff = export_onnx(model, fp32_path, image_size=metadata["image_size"])
        logger.info("Exported FP32 ONNX, max |torch - onnx| = %.3e", max_diff)

    # --- INT8 -------------------------------------------------------------
    int8_path = onnx_dir / "classifier_int8.onnx"
    if args.skip_int8:
        logger.info("Skipping INT8 export")
    elif int8_path.exists() and not args.force:
        logger.info("%s already exists (use --force to re-export)", int8_path)
    else:
        _export_int8(fp32_path, int8_path, labels["image_size"], args.calibration_samples)

    # --- Detection --------------------------------------------------------
    detector_path = onnx_dir / "detector_fp32.onnx"
    if args.with_detection and (not detector_path.exists() or args.force):
        from models.optimisation.export_detection import (
            export_detection_onnx,
            write_detection_metadata,
        )

        metadata = export_detection_onnx(detector_path)
        write_detection_metadata(artifacts_dir, metadata)
        logger.info(
            "Exported detector: %d classes, max |torch - onnx| = %.3e",
            metadata["num_classes"],
            metadata["max_abs_diff"],
        )
    elif detector_path.exists():
        logger.info("%s already exists", detector_path)

    # --- Detection --------------------------------------------------------
    detector_path = onnx_dir / "detector_fp32.onnx"
    if args.with_detection and (not detector_path.exists() or args.force):
        from models.optimisation.export_detection import (
            export_detection_onnx,
            write_detection_metadata,
        )

        metadata = export_detection_onnx(detector_path)
        write_detection_metadata(artifacts_dir, metadata)
        logger.info(
            "Exported detector: %d classes, max |torch - onnx| = %.3e",
            metadata["num_classes"],
            metadata["max_abs_diff"],
        )
    elif detector_path.exists():
        logger.info("%s already exists", detector_path)

    logger.info("Artifacts ready in %s", artifacts_dir)
    for path in sorted(artifacts_dir.rglob("*")):
        if path.is_file() and "runs" not in path.parts:
            logger.info(
                "  %-40s %6.1f MB", path.relative_to(artifacts_dir), path.stat().st_size / 1e6
            )

    return 0


def _export_int8(
    fp32_path: Path, int8_path: Path, image_size: int, calibration_samples: int
) -> None:
    """Quantize to INT8, warning about the measured accuracy cost.

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


if __name__ == "__main__":
    sys.exit(main())
