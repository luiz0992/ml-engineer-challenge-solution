#!/usr/bin/env python3
"""Full model evaluation: per-class accuracy, calibration, and retrieval quality.

Produces the measurements the model cards would otherwise have to leave
unstated. Three evaluations, each closing a gap that aggregate accuracy hides:

**Per-class accuracy.** An 85.88% average says nothing about the worst class.
Fine-grained categories are usually far weaker, and a caller relying on the
headline figure for a specific category may be badly misled.

**Calibration.** Whether a reported 0.9 probability means the model is right
90% of the time. Label smoothing and MixUp generally improve calibration, but
"generally" is not a measurement. Expected Calibration Error and a reliability
table are computed, because an API returning probabilities implicitly invites
callers to threshold on them.

**Retrieval precision across all classes**, not the five sampled by hand.

Usage::

    uv run python scripts/evaluate_models.py
    uv run python scripts/evaluate_models.py --classifier-samples 10000
    uv run python scripts/evaluate_models.py --skip-similarity
    uv run python scripts/evaluate_models.py --skip-classifier
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from api.services.runtime import create_session

logger = logging.getLogger("evaluate_models")

#: Bins for the reliability table. Ten is conventional and keeps each bin
#: populated enough for its accuracy estimate to mean something.
CALIBRATION_BINS = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--classifier-samples",
        type=int,
        default=10_000,
        help="Validation images to evaluate. 10000 is the full split.",
    )
    parser.add_argument(
        "--similarity-queries",
        type=int,
        default=1_000,
        help="Queries for retrieval evaluation, sampled across all classes",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Reuse the classifier section already in --output and only re-measure retrieval.",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/evaluation.json"))
    return parser


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def evaluate_classifier(
    artifacts_dir: Path, data_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Measure per-class accuracy and calibration on the validation split."""

    from api.utils.image_processing import (
        PreprocessConfig,
        preprocess_image,
        softmax,
        stack_batch,
    )
    from models.data.tiny_imagenet import resolve_root

    labels = json.loads((artifacts_dir / "labels.json").read_text())
    class_names, wnids = labels["class_names"], labels["wnids"]
    wnid_to_index = {w: i for i, w in enumerate(wnids)}

    session, input_name = create_session(
        str(artifacts_dir / "onnx" / "classifier_fp32.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    root = resolve_root(data_dir)
    files = sorted((root / "val").glob("*/*.JPEG"))[: args.classifier_samples]
    logger.info("Evaluating the classifier on %d images", len(files))

    config = PreprocessConfig(image_size=224)
    confidences: list[float] = []
    correct_flags: list[bool] = []
    per_class: dict[int, list[bool]] = defaultdict(list)
    top5_correct = 0

    for start in range(0, len(files), args.batch_size):
        chunk = files[start : start + args.batch_size]
        batch = stack_batch([preprocess_image(p.read_bytes(), config) for p in chunk])
        probabilities = softmax(session.run(None, {input_name: batch})[0])

        truth = np.array([wnid_to_index[p.parent.name] for p in chunk])
        predicted = probabilities.argmax(axis=-1)
        top5 = np.argsort(-probabilities, axis=-1)[:, :5]

        for i, label in enumerate(truth):
            is_correct = bool(predicted[i] == label)
            per_class[int(label)].append(is_correct)
            correct_flags.append(is_correct)
            confidences.append(float(probabilities[i, predicted[i]]))
            top5_correct += int(label in top5[i])

        if start % (args.batch_size * 40) == 0:
            logger.info("  %d / %d", start, len(files))

    accuracy = float(np.mean(correct_flags))
    class_accuracies = {class_names[c]: float(np.mean(flags)) for c, flags in per_class.items()}
    ranked = sorted(class_accuracies.items(), key=lambda kv: kv[1])

    calibration = compute_calibration(np.array(confidences), np.array(correct_flags, dtype=bool))

    return {
        "num_samples": len(files),
        "acc_top1": round(accuracy, 4),
        "acc_top5": round(top5_correct / len(files), 4),
        "per_class": {
            "num_classes": len(class_accuracies),
            "mean": round(float(np.mean(list(class_accuracies.values()))), 4),
            "std": round(float(np.std(list(class_accuracies.values()))), 4),
            "min": round(ranked[0][1], 4),
            "max": round(ranked[-1][1], 4),
            "worst_10": [{"class": n, "accuracy": round(a, 4)} for n, a in ranked[:10]],
            "best_10": [{"class": n, "accuracy": round(a, 4)} for n, a in reversed(ranked[-10:])],
            "below_50_percent": [n for n, a in ranked if a < 0.5],
        },
        "calibration": calibration,
    }


def compute_calibration(
    confidences: np.ndarray, correct: np.ndarray, bins: int = CALIBRATION_BINS
) -> dict[str, Any]:
    """Expected Calibration Error and a reliability table.

    ECE is the weighted mean gap between confidence and accuracy across
    equal-width confidence bins. A perfectly calibrated model scores 0: among
    predictions made with 70% confidence, exactly 70% are correct.

    The sign of the gap matters as much as its size. Confidence above accuracy
    is *overconfidence*, which is the dangerous direction — a caller
    thresholding at 0.9 gets fewer correct answers than they expect.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidences)

    ece = 0.0
    table: list[dict[str, Any]] = []

    for lower, upper in pairwise(edges):
        # Upper-inclusive on the final bin so confidence 1.0 is counted.
        in_bin = (confidences > lower) & (confidences <= upper)
        if upper == edges[-1]:
            in_bin |= confidences == lower

        count = int(in_bin.sum())
        if count == 0:
            continue

        bin_confidence = float(confidences[in_bin].mean())
        bin_accuracy = float(correct[in_bin].mean())
        ece += (count / total) * abs(bin_confidence - bin_accuracy)

        table.append(
            {
                "range": f"{lower:.1f}-{upper:.1f}",
                "count": count,
                "mean_confidence": round(bin_confidence, 4),
                "accuracy": round(bin_accuracy, 4),
                "gap": round(bin_accuracy - bin_confidence, 4),
            }
        )

    mean_confidence = float(confidences.mean())
    mean_accuracy = float(correct.mean())

    return {
        "expected_calibration_error": round(ece, 4),
        "mean_confidence": round(mean_confidence, 4),
        "mean_accuracy": round(mean_accuracy, 4),
        # Positive means the model claims more confidence than it earns.
        "overconfidence": round(mean_confidence - mean_accuracy, 4),
        "bins": table,
    }


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def evaluate_similarity(
    artifacts_dir: Path, data_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Measure retrieval precision across all classes.

    Queries come from the *validation* split while the index holds training
    images, so a query can never retrieve itself and inflate the result.
    """
    import faiss

    from api.utils.image_processing import PreprocessConfig, preprocess_image, stack_batch
    from models.data.tiny_imagenet import resolve_root

    index_path = artifacts_dir / "similarity.index"
    if not index_path.exists():
        logger.warning("No similarity index; skipping retrieval evaluation")
        return {}

    labels = json.loads((artifacts_dir / "labels.json").read_text())
    class_names, wnids = labels["class_names"], labels["wnids"]
    wnid_to_index = {w: i for i, w in enumerate(wnids)}

    manifest = json.loads((artifacts_dir / "similarity_manifest.json").read_text())
    index_labels = np.array(manifest["labels"])
    index = faiss.read_index(str(index_path))

    session, input_name = create_session(
        str(artifacts_dir / "onnx" / "embedder_fp32.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    root = resolve_root(data_dir)
    all_files = sorted((root / "val").glob("*/*.JPEG"))
    rng = np.random.default_rng(0)
    chosen = rng.choice(
        len(all_files), size=min(args.similarity_queries, len(all_files)), replace=False
    )
    files = [all_files[i] for i in sorted(chosen)]

    logger.info("Evaluating retrieval on %d queries", len(files))

    config = PreprocessConfig(image_size=224)
    # Floats: each query contributes a fraction (hits / k), not a count.
    precision_at: dict[int, float] = {1: 0.0, 5: 0.0, 10: 0.0}
    per_class_hits: dict[int, list[float]] = defaultdict(list)

    for start in range(0, len(files), args.batch_size):
        chunk = files[start : start + args.batch_size]
        batch = stack_batch([preprocess_image(p.read_bytes(), config) for p in chunk])
        embeddings = np.ascontiguousarray(
            session.run(None, {input_name: batch})[0].astype(np.float32)
        )
        _, neighbours = index.search(embeddings, 10)

        for i, path in enumerate(chunk):
            truth = wnid_to_index[path.parent.name]
            retrieved = index_labels[neighbours[i]]
            for k in precision_at:
                precision_at[k] += int((retrieved[:k] == truth).sum()) / k
            per_class_hits[truth].append(float((retrieved[:5] == truth).sum()) / 5)

        if start % (args.batch_size * 40) == 0:
            logger.info("  %d / %d", start, len(files))

    per_class = {class_names[c]: float(np.mean(hits)) for c, hits in per_class_hits.items()}
    ranked = sorted(per_class.items(), key=lambda kv: kv[1])

    return {
        "num_queries": len(files),
        "index_size": int(index.ntotal),
        "precision_at_1": round(precision_at[1] / len(files), 4),
        "precision_at_5": round(precision_at[5] / len(files), 4),
        "precision_at_10": round(precision_at[10] / len(files), 4),
        "per_class": {
            "classes_evaluated": len(per_class),
            "mean": round(float(np.mean(list(per_class.values()))), 4),
            "std": round(float(np.std(list(per_class.values()))), 4),
            "worst_10": [{"class": n, "precision_at_5": round(p, 4)} for n, p in ranked[:10]],
            "best_10": [
                {"class": n, "precision_at_5": round(p, 4)} for n, p in reversed(ranked[-10:])
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    report: dict[str, Any] = {}
    if args.output.exists():
        try:
            report = json.loads(args.output.read_text())
        except json.JSONDecodeError:
            report = {}

    if not args.skip_classifier:
        report["classifier"] = evaluate_classifier(args.artifacts_dir, args.data_dir, args)
    elif "classifier" not in report:
        raise SystemExit(
            "No classifier section in the existing report. Re-run without --skip-classifier."
        )

    if not args.skip_similarity:
        similarity = evaluate_similarity(args.artifacts_dir, args.data_dir, args)
        if similarity:
            report["similarity"] = similarity

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    _print_summary(report)
    logger.info("Wrote %s", args.output)
    return 0


def _print_summary(report: dict[str, Any]) -> None:
    classifier = report["classifier"]
    print(f"\n{'=' * 70}\nCLASSIFIER ({classifier['num_samples']} images)\n{'=' * 70}")
    print(
        f"  top-1 {classifier['acc_top1'] * 100:.2f}%   top-5 {classifier['acc_top5'] * 100:.2f}%"
    )

    per_class = classifier["per_class"]
    print(
        f"\n  Per-class accuracy over {per_class['num_classes']} classes: "
        f"mean {per_class['mean'] * 100:.1f}%, sd {per_class['std'] * 100:.1f}pp, "
        f"range {per_class['min'] * 100:.0f}%-{per_class['max'] * 100:.0f}%"
    )
    print(f"  Classes below 50%: {len(per_class['below_50_percent'])}")
    print("  Weakest:")
    for entry in per_class["worst_10"][:5]:
        print(f"    {entry['class']:<28} {entry['accuracy'] * 100:5.1f}%")

    calibration = classifier["calibration"]
    print(
        f"\n  Calibration: ECE {calibration['expected_calibration_error']:.4f}, "
        f"mean confidence {calibration['mean_confidence']:.3f} vs accuracy "
        f"{calibration['mean_accuracy']:.3f} "
        f"({'over' if calibration['overconfidence'] > 0 else 'under'}confident by "
        f"{abs(calibration['overconfidence']):.3f})"
    )
    print(f"    {'range':<12}{'n':>7}{'confidence':>12}{'accuracy':>10}{'gap':>8}")
    for row in calibration["bins"]:
        print(
            f"    {row['range']:<12}{row['count']:>7}{row['mean_confidence']:>12.3f}"
            f"{row['accuracy']:>10.3f}{row['gap']:>8.3f}"
        )

    if "similarity" in report:
        similarity = report["similarity"]
        print(f"\n{'=' * 70}\nSIMILARITY ({similarity['num_queries']} queries)\n{'=' * 70}")
        print(
            f"  P@1 {similarity['precision_at_1'] * 100:.1f}%   "
            f"P@5 {similarity['precision_at_5'] * 100:.1f}%   "
            f"P@10 {similarity['precision_at_10'] * 100:.1f}%"
        )
        sim_per_class = similarity["per_class"]
        print(
            f"  Per-class P@5 over {sim_per_class['classes_evaluated']} classes: "
            f"mean {sim_per_class['mean'] * 100:.1f}%, sd {sim_per_class['std'] * 100:.1f}pp"
        )
        print("  Weakest:")
        for entry in sim_per_class["worst_10"][:5]:
            print(f"    {entry['class']:<28} {entry['precision_at_5'] * 100:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
