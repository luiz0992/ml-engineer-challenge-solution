#!/usr/bin/env python3
"""Evaluate the object detector on COCO val2017.

Computes mAP with ``pycocotools``, the reference implementation used by the
COCO leaderboard, rather than a reimplementation. Detection mAP has enough
subtleties — IoU thresholds, area ranges, the 101-point interpolated
precision-recall curve, crowd handling — that a hand-rolled version is far more
likely to be subtly wrong than to be useful.

Until this runs, the detector's accuracy on this deployment is simply
unmeasured, and the published COCO figure is a claim about someone else's
evaluation.

Two details that would silently corrupt the result:

**Category IDs are not class indices.** COCO's 80 categories carry
non-contiguous IDs from 1 to 90. The model outputs a dense index in 0..79.
Submitting the index as a category ID scores almost everything as a mismatch
and produces a plausible-looking but near-zero mAP.

**Boxes must be in ``[x, y, width, height]``**, not corner coordinates. COCO
expects the former; the API returns the latter because that is what clients
overlay. Submitting corners silently halves the apparent box sizes.

Usage::

    uv run python scripts/evaluate_detector.py
    uv run python scripts/evaluate_detector.py --limit 500 --score-threshold 0.05
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from api.services.runtime import create_session

logger = logging.getLogger("evaluate_detector")

#: Low by design. mAP integrates precision over the full recall curve, so
#: discarding low-scoring detections truncates the curve and *understates* the
#: score. The serving default of 0.5 is an operating point for a user; it is the
#: wrong threshold for measuring the model.
DEFAULT_SCORE_THRESHOLD = 0.01


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="instances_val2017.json (located automatically if omitted)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Images to evaluate; 0 = all")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/detection_eval.json"))
    return parser


def locate_coco(data_dir: Path, annotations: Path | None) -> tuple[Path, Path]:
    """Find the COCO images and annotation file."""
    if annotations and annotations.exists():
        image_dirs = list(annotations.parent.parent.rglob("val2017"))
        if image_dirs:
            return image_dirs[0], annotations

    candidates = list(data_dir.rglob("instances_val2017.json"))
    if not candidates:
        raise SystemExit(
            "COCO annotations not found. Download them with:\n"
            "  python scripts/setup/download_datasets.py --dataset coco_sample"
        )

    annotation_path = candidates[0]
    image_dirs = [p for p in data_dir.rglob("val2017") if p.is_dir()]
    if not image_dirs:
        raise SystemExit(f"Found {annotation_path} but no val2017 image directory")

    return image_dirs[0], annotation_path


def run_detections(
    artifacts_dir: Path,
    image_dir: Path,
    image_records: list[dict[str, Any]],
    index_to_category: dict[int, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run the detector over the evaluation images.

    Returns detections in COCO result format.
    """

    from api.utils.image_processing import boxes_to_absolute, preprocess_for_detection

    session, input_name = create_session(
        str(artifacts_dir / "onnx" / "detector_fp32.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    results: list[dict[str, Any]] = []

    for start in range(0, len(image_records), args.batch_size):
        chunk = image_records[start : start + args.batch_size]

        arrays, sizes, ids = [], [], []
        for record in chunk:
            path = image_dir / record["file_name"]
            if not path.exists():
                continue
            array, width, height = preprocess_for_detection(path.read_bytes())
            arrays.append(array)
            sizes.append((width, height))
            ids.append(record["id"])

        if not arrays:
            continue

        scores, boxes = session.run(None, {input_name: np.stack(arrays)})

        for i, image_id in enumerate(ids):
            width, height = sizes[i]
            best_class = scores[i].argmax(axis=-1)
            best_score = scores[i].max(axis=-1)

            keep = np.where(best_score >= args.score_threshold)[0]
            if keep.size == 0:
                continue

            absolute = boxes_to_absolute(boxes[i][keep], width, height)

            for query, box in zip(keep, absolute, strict=True):
                class_index = int(best_class[query])
                x_min, y_min, x_max, y_max = (float(v) for v in box)

                results.append(
                    {
                        "image_id": int(image_id),
                        # Dense index -> COCO category ID. Submitting the index
                        # directly scores nearly everything as a mismatch.
                        "category_id": index_to_category[class_index],
                        # COCO wants [x, y, w, h], not corners.
                        "bbox": [
                            round(x_min, 2),
                            round(y_min, 2),
                            round(x_max - x_min, 2),
                            round(y_max - y_min, 2),
                        ],
                        "score": round(float(best_score[query]), 5),
                    }
                )

        if start % (args.batch_size * 20) == 0:
            logger.info("  %d / %d images", start, len(image_records))

    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_dir, annotation_path = locate_coco(args.data_dir, args.annotations)
    logger.info("Images: %s", image_dir)
    logger.info("Annotations: %s", annotation_path)

    # pycocotools prints a banner on load; suppress it so the output is legible.
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation_path))

    # The model's dense output index maps to COCO's non-contiguous category IDs
    # in sorted order, which is how the 80-class label list is conventionally
    # built.
    category_ids = sorted(coco.getCatIds())
    index_to_category = dict(enumerate(category_ids))

    metadata = json.loads((args.artifacts_dir / "detection_labels.json").read_text())
    if len(category_ids) != metadata["num_classes"]:
        raise SystemExit(
            f"COCO has {len(category_ids)} categories but the model declares "
            f"{metadata['num_classes']}; the index mapping would be wrong."
        )

    # Only images actually present on disk: the download script subsamples, so
    # evaluating against the full annotation set would count every absent image
    # as a complete miss and understate mAP dramatically.
    available = {p.name for p in image_dir.glob("*.jpg")}
    image_records = [
        record for record in coco.loadImgs(coco.getImgIds()) if record["file_name"] in available
    ]
    if args.limit:
        image_records = image_records[: args.limit]

    logger.info(
        "Evaluating %d of %d annotated images present on disk",
        len(image_records),
        len(coco.getImgIds()),
    )
    if not image_records:
        raise SystemExit("No annotated images found on disk")

    detections = run_detections(
        args.artifacts_dir, image_dir, image_records, index_to_category, args
    )
    logger.info("Produced %d detections", len(detections))

    if not detections:
        raise SystemExit("The detector produced no detections above the threshold")

    with contextlib.redirect_stdout(io.StringIO()):
        coco_detections = coco.loadRes(detections)
        evaluator = COCOeval(coco, coco_detections, iouType="bbox")
        # Restrict to the images actually evaluated, or every unevaluated image
        # counts against recall.
        evaluator.params.imgIds = [r["id"] for r in image_records]
        evaluator.evaluate()
        evaluator.accumulate()

    summary = io.StringIO()
    with contextlib.redirect_stdout(summary):
        evaluator.summarize()

    stats = evaluator.stats
    report = {
        "num_images": len(image_records),
        "num_detections": len(detections),
        "score_threshold": args.score_threshold,
        "metrics": {
            "mAP@[.5:.95]": round(float(stats[0]), 4),
            "mAP@0.5": round(float(stats[1]), 4),
            "mAP@0.75": round(float(stats[2]), 4),
            "mAP_small": round(float(stats[3]), 4),
            "mAP_medium": round(float(stats[4]), 4),
            "mAP_large": round(float(stats[5]), 4),
            "AR@1": round(float(stats[6]), 4),
            "AR@10": round(float(stats[7]), 4),
            "AR@100": round(float(stats[8]), 4),
            "AR_small": round(float(stats[9]), 4),
            "AR_medium": round(float(stats[10]), 4),
            "AR_large": round(float(stats[11]), 4),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 70}\nDETECTOR — COCO val2017 ({len(image_records)} images)\n{'=' * 70}")
    for name, value in report["metrics"].items():
        print(f"  {name:<16} {value:.4f}")
    print(
        "\n  Small-object AP is the known weakness of this model family, and the "
        "\n  R18 backbone is weaker here than deeper variants."
    )
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
