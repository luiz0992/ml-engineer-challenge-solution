"""Run the full optimisation and benchmarking sweep.

Exports the models to ONNX, quantizes each to INT8, benchmarks every
available backend, and writes the reports the challenge asks for under
``benchmarks/``.

Usage::

    uv run python -m models.optimisation.run_benchmarks
    uv run python -m models.optimisation.run_benchmarks --batch-sizes 1 8 32 --iterations 200
    uv run python -m models.optimisation.run_benchmarks --skip-export   # reuse artefacts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from models.optimisation.benchmark import (
    BenchmarkReport,
    LatencyResult,
    benchmark_onnx,
    benchmark_torch,
    capture_environment,
    ensure_tensorrt_loadable,
    format_markdown_report,
    now_iso,
)
from models.optimisation.export import export_onnx, load_classifier_from_run

logger = logging.getLogger("run_benchmarks")


def find_latest_run(runs_dir: Path) -> Path:
    """Locate the most recent training run directory."""
    candidates = [p for p in runs_dir.glob("*") if (p / "config.json").is_file()]
    if not candidates:
        raise SystemExit(
            f"No training runs with a config.json found under {runs_dir}. "
            f"Train a model first:\n  uv run python -m models.training.train"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Training run to benchmark")
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("models/artifacts/runs"), help="Where runs live"
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts/onnx"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Datasets used for calibration and post-quantization measurement",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=256,
        help="Images used to calibrate INT8 activation ranges",
    )
    parser.add_argument(
        "--detector-calibration-samples",
        type=int,
        default=64,
        help=(
            "Calibration images for the detector. Lower than the default: "
            "percentile calibration holds a 2048-bin histogram per tensor, and "
            "RT-DETR's 640x640 activations are roughly eight times the "
            "classifier's, so 256 samples exhausts host memory."
        ),
    )
    parser.add_argument(
        "--quantize-only",
        action="store_true",
        help="Build the INT8 artefacts and exit without benchmarking",
    )
    parser.add_argument(
        "--accuracy-samples",
        type=int,
        default=2000,
        help="Validation images used to measure post-quantization accuracy (0 to skip)",
    )
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing ONNX artefacts")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--skip-tensorrt", action="store_true")
    parser.add_argument("--skip-compile", action="store_true", help="Skip torch.compile benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["classifier"],
        choices=["classifier", "detector", "embedder", "all"],
        help="Which models to benchmark. Detector and embedder are inference-only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    run_dir = args.run_dir or find_latest_run(args.runs_dir)
    logger.info("Benchmarking run: %s", run_dir)

    model, metadata = load_classifier_from_run(run_dir)
    image_size = metadata["image_size"]

    fp32_path = args.artifacts_dir / "classifier_fp32.onnx"
    int8_path = args.artifacts_dir / "classifier_int8.onnx"

    selected = set(args.models)
    if "all" in selected:
        selected = {"classifier", "detector", "embedder"}

    # --- Export -----------------------------------------------------------
    if not args.skip_export or not fp32_path.exists():
        max_diff = export_onnx(model, fp32_path, image_size=image_size)
        logger.info("ONNX export verified: max |torch - onnx| = %.3e", max_diff)

    # --- Quantize ---------------------------------------------------------
    # All three are built before anything is benchmarked. Calibration is the
    # most memory-hungry step in the sweep -- the detector collects 2048-bin
    # histograms over 640x640 activations -- and interleaving it with live
    # InferenceSessions pushed the process into the OOM killer. Doing the work
    # up front also means `--quantize-only` can stop here.
    fidelity: dict[str, dict[str, float]] = {}

    if not args.skip_int8:
        if not args.skip_export or not int8_path.exists():
            _try_quantize(
                "classifier", partial(_quantize_classifier, args, fp32_path, int8_path, image_size)
            )

        for name, image_size_for in (("detector", 640), ("embedder", image_size)):
            if name not in selected:
                continue
            aux_fp32 = args.artifacts_dir / f"{name}_fp32.onnx"
            aux_int8 = args.artifacts_dir / f"{name}_int8.onnx"
            if not aux_fp32.exists() or (args.skip_export and aux_int8.exists()):
                continue
            _try_quantize(
                name, partial(_quantize_auxiliary, args, name, aux_fp32, aux_int8, image_size_for)
            )

    if args.quantize_only:
        logger.info("Quantization complete; skipping the benchmark sweep (--quantize-only)")
        return 0

    # --- Measure the cost of INT8 -----------------------------------------
    # Deliberately independent of whether quantization just ran. Tying the two
    # together meant a `--skip-export` sweep reused the INT8 artefact but
    # reported no accuracy for it, which is how the committed report ended up
    # with a null accuracy column.
    accuracies: dict[str, dict[str, float]] = {}
    if args.accuracy_samples > 0 and int8_path.exists():
        # Guarded like every other dataset-dependent step. Decoupling the
        # measurement from the build means a `--skip-export` sweep now always
        # touches `data/`, so an artefacts-only host -- a GPU box with no
        # dataset -- would otherwise lose the whole report to a traceback
        # instead of one column.
        _try_quantize(
            "classifier accuracy",
            partial(
                _collect_classifier_accuracy, args, fp32_path, int8_path, image_size, accuracies
            ),
        )

    if args.accuracy_samples > 0 and "embedder" in selected:
        embedder_fp32 = args.artifacts_dir / "embedder_fp32.onnx"
        embedder_int8 = args.artifacts_dir / "embedder_int8.onnx"
        if embedder_fp32.exists() and embedder_int8.exists():
            _try_quantize(
                "embedder fidelity",
                partial(
                    _measure_embedder_fidelity,
                    args,
                    embedder_fp32,
                    embedder_int8,
                    image_size,
                    fidelity,
                ),
            )

    # --- Benchmark --------------------------------------------------------
    # Both must run before any InferenceSession is created. CUDA resolves cuDNN
    # lazily at the first kernel launch, so without the preload a session builds
    # cleanly and then fails every inference with NOT_IMPLEMENTED. TensorRT
    # fails earlier but silently, by falling back to CPU.
    from api.services.runtime import preload_cuda_libraries

    preload_cuda_libraries()
    trt_available = (not args.skip_tensorrt) and ensure_tensorrt_loadable()

    report = BenchmarkReport(environment=capture_environment(), generated_at=now_iso())
    has_cuda = report.environment.get("cuda_available", False)

    for batch_size in args.batch_sizes:
        logger.info("--- batch size %d ---", batch_size)
        common = {
            "batch_size": batch_size,
            "image_size": image_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
        }

        # partial binds the current loop values eagerly. Lambdas would capture
        # `common` by reference, so if these calls were ever deferred every
        # entry would silently use the final batch size.
        cases: list[tuple[str, Callable[[], LatencyResult]]] = [
            ("pytorch cpu", partial(benchmark_torch, model, device="cpu", **common))
        ]

        if has_cuda:
            cases += [
                ("pytorch cuda", partial(benchmark_torch, model, device="cuda", **common)),
                (
                    "pytorch cuda bf16",
                    partial(benchmark_torch, model, device="cuda", precision="bf16", **common),
                ),
            ]
            if not args.skip_compile:
                cases.append(
                    (
                        "pytorch compiled",
                        partial(
                            benchmark_torch, model, device="cuda", compile_model=True, **common
                        ),
                    )
                )

        cases.append(
            (
                "onnx cpu fp32",
                partial(benchmark_onnx, fp32_path, provider="CPUExecutionProvider", **common),
            )
        )

        if int8_path.exists():
            cases.append(
                (
                    "onnx cpu int8",
                    partial(
                        benchmark_onnx,
                        int8_path,
                        provider="CPUExecutionProvider",
                        precision="int8",
                        **common,
                    ),
                )
            )

        if has_cuda:
            cases.append(
                (
                    "onnx cuda fp32",
                    partial(benchmark_onnx, fp32_path, provider="CUDAExecutionProvider", **common),
                )
            )
            if trt_available:
                cases.append(
                    (
                        "onnx tensorrt fp16",
                        partial(
                            benchmark_onnx,
                            fp32_path,
                            provider="TensorrtExecutionProvider",
                            precision="fp16",
                            **common,
                        ),
                    )
                )

        for label, case in cases:
            _try(report, label, case)

    # Attach measured accuracy to the ONNX rows, so the report presents speed
    # and accuracy side by side rather than leaving the reader to correlate two
    # separate tables.
    _annotate_accuracy(report, accuracies, args.accuracy_samples)

    # --- Additional models --------------------------------------------------
    for name, filename, image_size in (
        ("detector", "detector_fp32.onnx", 640),
        ("embedder", "embedder_fp32.onnx", 224),
    ):
        if name not in selected:
            continue

        path = args.artifacts_dir / filename
        if not path.exists():
            logger.warning("%s not found at %s; skipping", name, path)
            continue

        logger.info("--- %s ---", name)
        aux_int8 = args.artifacts_dir / f"{name}_int8.onnx"

        for batch_size in args.batch_sizes:
            # Detection at 640x640 is roughly eight times the pixels of a
            # 224x224 classification input, so large batches exhaust GPU memory
            # long before they saturate compute.
            if name == "detector" and batch_size > 8:
                continue

            common = {
                "batch_size": batch_size,
                "image_size": image_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
            }
            extra_cases: list[tuple[str, Callable[[], LatencyResult]]] = [
                (
                    f"{name} onnx cpu",
                    partial(benchmark_onnx, path, provider="CPUExecutionProvider", **common),
                )
            ]
            if aux_int8.exists():
                extra_cases.append(
                    (
                        f"{name} onnx cpu int8",
                        partial(
                            benchmark_onnx,
                            aux_int8,
                            provider="CPUExecutionProvider",
                            precision="int8",
                            **common,
                        ),
                    )
                )
            if has_cuda:
                extra_cases.append(
                    (
                        f"{name} onnx cuda",
                        partial(benchmark_onnx, path, provider="CUDAExecutionProvider", **common),
                    )
                )
                if trt_available:
                    extra_cases.append(
                        (
                            f"{name} tensorrt fp16",
                            partial(
                                benchmark_onnx,
                                path,
                                provider="TensorrtExecutionProvider",
                                precision="fp16",
                                **common,
                            ),
                        )
                    )

            for label, case in extra_cases:
                _try(report, label, case, model=name)

    # --- Report -----------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.output_dir / "results.json")
    (args.output_dir / "README.md").write_text(format_markdown_report(report))
    logger.info("Wrote %s", args.output_dir / "README.md")

    if fidelity or accuracies:
        quantization_path = args.output_dir / "quantization.json"
        quantization_path.write_text(
            json.dumps(_quantization_summary(args, accuracies, fidelity), indent=2) + "\n"
        )
        logger.info("Wrote %s", quantization_path)

    print("\n" + format_markdown_report(report))
    return 0


def _quantization_summary(
    args: argparse.Namespace,
    accuracies: dict[str, dict[str, float]],
    fidelity: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Collect what INT8 cost each model into one report.

    Each model is judged by a different metric -- top-1 for the classifier,
    retrieval agreement for the embedder, COCO mAP for the detector -- so this
    records the metric alongside the numbers rather than implying they are
    comparable. The detector's mAP needs `pycocotools` and annotations, so it
    is produced by `scripts/evaluate_detector.py` and referenced here instead of
    recomputed; a file named `quantization.json` that silently covered one of
    three models would be worse than no file.
    """
    summary: dict[str, Any] = {}

    if accuracies:
        summary["classifier"] = {
            "metric": "top-1 accuracy",
            "fp32": accuracies.get("fp32", {}).get("acc_top1"),
            "int8": accuracies.get("int8", {}).get("acc_top1"),
            "num_samples": accuracies.get("fp32", {}).get("num_samples"),
        }

    if "embedder" in fidelity:
        summary["embedder"] = {
            "metric": "recall@k against the FP32 index",
            **fidelity["embedder"],
        }

    detector_report = args.output_dir / "detection_eval_int8.json"
    summary["detector"] = {
        "metric": "COCO mAP@[.5:.95]",
        "source": str(detector_report),
        "note": (
            "Produced by scripts/evaluate_detector.py --model detector_int8.onnx; "
            "mAP requires annotations and pycocotools, so it is not computed here."
        ),
        "measured": detector_report.exists(),
    }

    return summary


def _try_quantize(label: str, build: Callable[[], None]) -> None:
    """Run one quantization step, recording a failure rather than aborting.

    A calibration failure should cost the INT8 rows of the report, not the
    whole sweep, exactly as a missing execution provider does.
    """
    try:
        build()
    except Exception as exc:
        logger.warning(
            "  %-22s SKIPPED: %s: %s", f"{label} int8", type(exc).__name__, str(exc)[:160]
        )


def _quantize_classifier(
    args: argparse.Namespace, fp32_path: Path, int8_path: Path, image_size: int
) -> None:
    """Quantize the classifier, calibrating on Tiny-ImageNet."""
    from models.data.augmentation import AugmentationConfig, build_eval_transform
    from models.data.tiny_imagenet import build_datasets
    from models.optimisation.quantize import TensorDatasetCalibrationReader, quantize_int8

    cfg = AugmentationConfig(image_size=image_size)
    data = build_datasets(str(args.data_dir), cfg)

    # Calibrate on training data with the *evaluation* transform: augmentation
    # randomness would produce activation ranges that do not reflect serving.
    calibration_set = data.train
    calibration_set.transform = build_eval_transform(cfg)

    reader = TensorDatasetCalibrationReader(
        calibration_set, input_name="images", num_samples=args.calibration_samples
    )
    quantize_int8(fp32_path, int8_path, reader)


def _quantize_auxiliary(
    args: argparse.Namespace, name: str, fp32_path: Path, int8_path: Path, image_size: int
) -> None:
    """Quantize the detector or the embedder.

    The two differ only in where calibration data comes from. The embedder
    shares the classifier's backbone and input pipeline, so it reuses
    Tiny-ImageNet. The detector has no torch ``Dataset`` behind it and is
    calibrated on COCO through ``preprocess_for_detection`` -- the same
    function the API calls per request, which matters because RT-DETR skips
    mean/std normalisation and resizes without preserving aspect ratio.

    Neither is *measured* here; see ``_measure_embedder_fidelity`` and, for the
    detector's COCO mAP, ``scripts/evaluate_detector.py --model``.
    """
    from models.optimisation.quantize import quantize_int8

    if name == "embedder":
        _quantize_classifier(args, fp32_path, int8_path, image_size)
        return

    from api.utils.image_processing import preprocess_for_detection
    from models.optimisation.quantize import ImageFileCalibrationReader

    image_dir = args.data_dir / "coco_val2017" / "val2017"
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"No COCO images under {image_dir}. Download them with:\n"
            f"  python scripts/setup/download_datasets.py --dataset coco_sample"
        )

    reader = ImageFileCalibrationReader(
        images,
        input_name="pixel_values",
        preprocess=lambda data: preprocess_for_detection(data)[0],
        num_samples=args.detector_calibration_samples,
    )
    quantize_int8(fp32_path, int8_path, reader)


def _collect_classifier_accuracy(
    args: argparse.Namespace,
    fp32_path: Path,
    int8_path: Path,
    image_size: int,
    accuracies: dict[str, dict[str, float]],
) -> None:
    """Measure accuracy into ``accuracies``, for use through `_try_quantize`."""
    accuracies.update(_measure_classifier_accuracy(args, fp32_path, int8_path, image_size))


def _measure_classifier_accuracy(
    args: argparse.Namespace, fp32_path: Path, int8_path: Path, image_size: int
) -> dict[str, dict[str, float]]:
    """Measure top-1/top-5 for both precisions.

    A speedup figure without its accuracy cost is not enough to make a
    deployment decision, so the report carries both side by side.
    """
    from models.data.augmentation import AugmentationConfig
    from models.data.tiny_imagenet import build_datasets
    from models.optimisation.quantize import evaluate_onnx_accuracy

    data = build_datasets(str(args.data_dir), AugmentationConfig(image_size=image_size))

    accuracies: dict[str, dict[str, float]] = {}
    for label, path in (("fp32", fp32_path), ("int8", int8_path)):
        metrics = evaluate_onnx_accuracy(path, data.val, num_samples=args.accuracy_samples)
        accuracies[label] = metrics
        logger.info(
            "classifier %s accuracy: top1=%.2f%% top5=%.2f%% (n=%d)",
            label.upper(),
            metrics["acc_top1"] * 100,
            metrics["acc_top5"] * 100,
            metrics["num_samples"],
        )
    return accuracies


def _measure_embedder_fidelity(
    args: argparse.Namespace,
    fp32_path: Path,
    int8_path: Path,
    image_size: int,
    fidelity: dict[str, dict[str, float]],
) -> None:
    """Measure whether the embedder's vector space survived quantization.

    See ``evaluate_embedding_fidelity`` for why top-1 accuracy does not apply
    to a model with no classifier head.
    """
    from models.data.augmentation import AugmentationConfig
    from models.data.tiny_imagenet import build_datasets
    from models.optimisation.quantize import evaluate_embedding_fidelity

    data = build_datasets(str(args.data_dir), AugmentationConfig(image_size=image_size))
    metrics = evaluate_embedding_fidelity(
        fp32_path, int8_path, data.val, num_samples=args.accuracy_samples
    )
    logger.info(
        "embedder INT8 fidelity: cosine=%.4f recall@%d=%.4f (n=%d)",
        metrics["mean_cosine_similarity"],
        int(metrics["top_k"]),
        metrics["recall_at_k"],
        int(metrics["num_samples"]),
    )
    fidelity["embedder"] = metrics


def _annotate_accuracy(
    report: BenchmarkReport, accuracies: dict[str, dict[str, float]], num_samples: int
) -> None:
    """Attach accuracy measurements to the corresponding benchmark rows."""
    if not accuracies:
        return

    baseline = accuracies.get("fp32", {}).get("acc_top1")
    seen: set[str] = set()

    for result in report.results:
        metrics = accuracies.get(result.precision)
        if metrics is None or not result.backend.startswith("onnxruntime"):
            continue

        # One row per precision is enough; the model is identical across batch
        # sizes, so repeating it would imply measurements that were not made.
        key = f"{result.backend}:{result.precision}"
        if key in seen:
            continue
        seen.add(key)

        result.accuracy_top1 = metrics["acc_top1"]
        note = f"top-5 {metrics['acc_top5'] * 100:.2f}%, n={int(num_samples)}"
        if baseline is not None and result.precision != "fp32":
            note += f", {(metrics['acc_top1'] - baseline) * 100:+.2f}pp vs FP32"
        result.notes = note


def _try(
    report: BenchmarkReport,
    label: str,
    fn: Callable[[], LatencyResult],
    *,
    model: str = "classifier",
) -> None:
    """Run one benchmark, recording failures without aborting the sweep.

    A missing execution provider or an unbuildable TensorRT engine should cost
    one row of the table, not the whole report.
    """
    try:
        result = fn()
        result.model = model
        report.results.append(result)
        logger.info(
            "  %-22s %7.2f ms  (p95 %6.2f)  %8.1f img/s",
            label,
            result.mean_ms,
            result.p95_ms,
            result.throughput_ips,
        )
    except Exception as exc:
        logger.warning("  %-22s SKIPPED: %s: %s", label, type(exc).__name__, str(exc)[:120])


if __name__ == "__main__":
    sys.exit(main())
