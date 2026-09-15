#!/usr/bin/env python3
"""Memory profiling for the serving path.

The performance suite already guards against a *leak* — RSS must not grow
across repeated inference. This answers the different question of where memory
actually goes, which a pass/fail threshold cannot:

**Resident footprint per model.** What each loaded model costs, measured as the
RSS delta across its load. This is the number that sizes a container: the
Compose and Kubernetes memory limits are only defensible if someone measured
them.

**Peak allocation per request.** Python-level allocation, via `tracemalloc`,
for one classification. Distinct from RSS, which the allocator holds onto after
a spike; peak allocation is what determines how much concurrency a memory limit
permits.

**Where it is allocated.** The top allocating source locations, so a
regression points at a line rather than at a number.

**Scaling with input size.** Whether a 2048x2048 upload costs proportionally
more than a 224x224 one. It should not: images are resized early, and a peak
that tracks the *upload* size means a full-resolution copy is being retained.

Usage::

    uv run python scripts/profile_memory.py
    uv run python scripts/profile_memory.py --requests 100 --output benchmarks/memory.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import io
import json
import logging
import secrets
import sys
import tracemalloc
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger("profile_memory")

#: Source files whose allocations are attributed to the service rather than to
#: the interpreter or a third-party library, for the "where" breakdown.
PROJECT_MARKERS = ("/api/", "/models/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/memory.json"))
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Inferences to average peak allocation over",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Allocating source locations to report",
    )
    return parser


def make_image(width: int, height: int) -> bytes:
    """A JPEG of the requested size, with enough detail to resist compression."""
    import numpy as np

    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype="uint8")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def warm_runtime(artifacts_dir: Path) -> float:
    """Charge one-time runtime initialisation to its own line item.

    ONNX Runtime defers three costs to first use: provider resolution at
    session creation; the CUDA context and cuDNN kernels, which are
    process-wide; and the allocation arena, which is not created until
    inference actually runs.

    None of them belongs to a model, but without this step all three land on
    whichever model is loaded and probed first. That made the classifier appear
    to cost 728 MB against the embedder's 88 MB, despite the two sharing a
    backbone and differing in artefact size by 0.1 MB.

    Two details matter. The probe must actually run, not merely be created, or
    the arena is still charged elsewhere. And it must use the *same providers*
    the service will use -- warming on CPU leaves the entire CUDA context to be
    paid by the first real model.
    """
    import numpy as np
    import onnxruntime as ort

    from api.services.runtime import create_session

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    gc.collect()
    before = rss_mb()
    session, input_name = create_session(
        str(artifacts_dir / "onnx" / "classifier_fp32.onnx"), providers=providers
    )
    session.run(None, {input_name: np.zeros((1, 3, 224, 224), dtype=np.float32)})
    cost = rss_mb() - before
    del session
    gc.collect()

    logger.info("%-11s +%.1f MB resident (one-time)", "runtime", cost)
    return round(cost, 1)


def rss_mb() -> float:
    """Resident set size in MB."""
    import psutil

    return float(psutil.Process().memory_info().rss) / 1024 / 1024


def profile_model_loading(artifacts_dir: Path) -> tuple[dict[str, Any], Any, Any]:
    """Measure the resident cost of loading each model.

    A garbage collection before each measurement keeps the previous model's
    garbage from being attributed to the next one.

    Returns the measurements together with the loaded service and its settings,
    so the request profiles below reuse these models rather than paying to load
    them a second time.
    """
    from api.config import AppEnv, InferenceBackend, Settings
    from api.services.model_service import ModelService

    settings = Settings(
        _env_file=None,
        app_env=AppEnv.DEVELOPMENT,
        # Generated, not hardcoded: this process signs nothing, and a literal
        # here would be a committed credential whatever the comment said.
        jwt_secret_key=secrets.token_hex(32),
        artifacts_dir=artifacts_dir,
        inference_backend=InferenceBackend.ONNX,
    )
    service = ModelService(settings)

    runtime_mb = warm_runtime(artifacts_dir)

    gc.collect()
    baseline = rss_mb()
    measurements: dict[str, Any] = {
        "baseline_rss_mb": round(baseline, 1),
        "runtime_init_mb": runtime_mb,
        "models": {},
    }

    for name, load in (
        ("classifier", service.load_classifier),
        ("detector", service.load_detector),
        ("embedder", service.load_embedder),
    ):
        gc.collect()
        before = rss_mb()
        try:
            model = load()
        except Exception as exc:
            logger.warning("Could not load %s: %s", name, exc)
            continue
        gc.collect()

        measurements["models"][name] = {
            "rss_delta_mb": round(rss_mb() - before, 1),
            "backend": model.backend.value,
            # Recorded so the report says which execution provider these
            # numbers describe. A CPU-only host has no CUDA context to pay
            # for and the figures are not comparable.
            "provider": model.session.get_providers()[0],
            "version": model.version,
        }
        logger.info(
            "%-11s +%.1f MB resident (%s)",
            name,
            measurements["models"][name]["rss_delta_mb"],
            model.backend.value,
        )

    measurements["total_rss_mb"] = round(rss_mb(), 1)
    return measurements, service, settings


async def profile_requests(
    service: Any, settings: Any, *, requests: int, top: int
) -> dict[str, Any]:
    """Measure peak allocation per request and attribute it to source lines."""
    from api.services.cache_service import CacheService
    from api.services.inference_service import InferenceService

    inference = InferenceService(service, CacheService(None), settings, None)
    image = make_image(512, 512)

    # Warm up: first-call allocations (lazy imports, ONNX arenas) are one-off
    # and would otherwise be charged to the first measured request.
    for _ in range(5):
        await inference.classify(image, correlation_id="warmup", use_cache=False)

    gc.collect()
    tracemalloc.start(25)
    rss_before = rss_mb()

    peaks: list[float] = []
    for index in range(requests):
        tracemalloc.reset_peak()
        await inference.classify(image, correlation_id=f"profile-{index}", use_cache=False)
        _, peak = tracemalloc.get_traced_memory()
        peaks.append(peak / 1024 / 1024)

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    gc.collect()

    peaks.sort()
    hot: list[dict[str, Any]] = []
    for stat in snapshot.statistics("lineno")[: top * 4]:
        frame = stat.traceback[0]
        if not any(marker in frame.filename for marker in PROJECT_MARKERS):
            continue
        hot.append(
            {
                "location": f"{Path(frame.filename).name}:{frame.lineno}",
                "size_kb": round(stat.size / 1024, 1),
                "count": stat.count,
            }
        )
        if len(hot) >= top:
            break

    return {
        "requests": requests,
        "peak_alloc_mb": {
            "mean": round(sum(peaks) / len(peaks), 3),
            "p50": round(peaks[len(peaks) // 2], 3),
            "p95": round(peaks[int(len(peaks) * 0.95)], 3),
            "max": round(peaks[-1], 3),
        },
        "rss_growth_mb": round(rss_mb() - rss_before, 1),
        "top_allocations": hot,
    }


async def profile_input_scaling(service: Any, settings: Any) -> list[dict[str, Any]]:
    """Measure peak allocation against upload size.

    Peak should stay roughly flat: the image is resized to 224x224 early, so a
    peak that tracks the upload means a full-resolution array is being held.
    """
    from api.services.cache_service import CacheService
    from api.services.inference_service import InferenceService

    inference = InferenceService(service, CacheService(None), settings, None)
    rows: list[dict[str, Any]] = []

    for edge in (224, 512, 1024, 2048):
        image = make_image(edge, edge)
        await inference.classify(image, correlation_id="warmup", use_cache=False)

        gc.collect()
        tracemalloc.start(1)
        tracemalloc.reset_peak()
        await inference.classify(image, correlation_id=f"scale-{edge}", use_cache=False)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows.append(
            {
                "edge_px": edge,
                "upload_kb": round(len(image) / 1024, 1),
                "peak_alloc_mb": round(peak / 1024 / 1024, 3),
            }
        )
        logger.info(
            "%5d px  upload %7.1f KB  peak %.3f MB",
            edge,
            rows[-1]["upload_kb"],
            rows[-1]["peak_alloc_mb"],
        )

    return rows


def format_report(report: dict[str, Any]) -> str:
    """Render the measurements as markdown."""
    lines = ["# Memory profile", ""]

    loading = report["loading"]
    lines += [
        f"ONNX Runtime initialisation costs **{loading['runtime_init_mb']} MB** once, at "
        f"first session creation, independently of which model triggers it. Measured "
        f"separately below so the per-model figures are comparable.",
        "",
        f"Resident after initialisation: **{loading['baseline_rss_mb']} MB**; "
        f"**{loading['total_rss_mb']} MB** with all three models loaded.",
        "",
        "## Resident cost per model",
        "",
        "| Model | Backend | Provider | RSS delta |",
        "| --- | --- | --- | ---: |",
    ]
    for name, entry in loading["models"].items():
        lines.append(
            f"| {name} | {entry['backend']} | {entry['provider']} | {entry['rss_delta_mb']} MB |"
        )

    peaks = report["requests"]["peak_alloc_mb"]
    lines += [
        "",
        "## Peak allocation per classification",
        "",
        f"Averaged over {report['requests']['requests']} requests, 512x512 input.",
        "",
        "| Statistic | Peak allocation |",
        "| --- | ---: |",
        f"| mean | {peaks['mean']} MB |",
        f"| p50 | {peaks['p50']} MB |",
        f"| p95 | {peaks['p95']} MB |",
        f"| max | {peaks['max']} MB |",
        "",
        f"RSS growth across the run: **{report['requests']['rss_growth_mb']} MB**.",
        "",
        "## What each request retains",
        "",
        "Allocations still live after the request completed -- where a leak would "
        "appear. The transient peak above is dominated by decode and resize buffers "
        "that are freed before the response is sent.",
        "",
        "| Location | Retained | Blocks |",
        "| --- | ---: | ---: |",
    ]
    for entry in report["requests"]["top_allocations"]:
        lines.append(f"| `{entry['location']}` | {entry['size_kb']} KB | {entry['count']} |")

    lines += [
        "",
        "## Scaling with upload size",
        "",
        "| Input | Upload | Peak allocation |",
        "| --- | ---: | ---: |",
    ]
    for row in report["scaling"]:
        lines.append(
            f"| {row['edge_px']}x{row['edge_px']} | {row['upload_kb']} KB | "
            f"{row['peak_alloc_mb']} MB |"
        )

    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    loading, service, settings = profile_model_loading(args.artifacts_dir)
    return {
        "loading": loading,
        "requests": await profile_requests(service, settings, requests=args.requests, top=args.top),
        "scaling": await profile_input_scaling(service, settings),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not (args.artifacts_dir / "onnx").is_dir():
        raise SystemExit(
            f"No model artefacts under {args.artifacts_dir}. Build them with:\n"
            f"  uv run python scripts/prepare_artifacts.py"
        )

    if args.requests < 1:
        raise SystemExit("--requests must be at least 1")

    report = asyncio.run(run(args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(format_report(report))

    logger.info("Wrote %s and %s", args.output, markdown)
    print("\n" + format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
