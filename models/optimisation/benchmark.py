"""Latency and throughput benchmarking across inference backends.

Produces the performance comparison the challenge asks for: PyTorch (eager and
compiled) against ONNX Runtime on CPU, CUDA, and TensorRT execution providers,
in both FP32 and INT8.

Measurement methodology
-----------------------
Naive timing of GPU work is almost always wrong, so this harness is explicit
about three things:

* **Warmup.** The first inferences pay for lazy CUDA context creation, cuDNN
  algorithm selection, and — for TensorRT — engine building or cache
  deserialisation. Warmup iterations are discarded.
* **Synchronisation.** CUDA kernel launches are asynchronous. Without
  ``torch.cuda.synchronize`` before stopping the clock, you measure launch
  overhead rather than execution, which produces impossibly fast results.
* **Percentiles, not just means.** A mean latency hides the tail that
  determines whether an API meets its SLO. p50/p95/p99 are all reported.

Each configuration is timed over many iterations and reported with the standard
deviation, so a reader can judge whether two results actually differ.
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 100


@dataclass(slots=True)
class LatencyResult:
    """Timing statistics for one backend at one batch size."""

    backend: str
    precision: str
    batch_size: int
    device: str

    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_ips: float

    iterations: int
    warmup: int
    #: Which model produced this measurement. Benchmarks now cover three, and a
    #: table mixing them without this column would be meaningless.
    model: str = "classifier"
    model_size_mb: float | None = None
    accuracy_top1: float | None = None
    notes: str = ""

    @property
    def per_image_ms(self) -> float:
        return self.mean_ms / self.batch_size


@dataclass(slots=True)
class BenchmarkReport:
    """A full benchmark sweep plus the environment it was measured in.

    Environment capture is not decoration: latency numbers are meaningless
    without the hardware, driver, and library versions that produced them, and
    a report that omits them cannot be reproduced or compared later.
    """

    results: list[LatencyResult] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "generated_at": self.generated_at,
                    "environment": self.environment,
                    "results": [asdict(r) for r in self.results],
                },
                indent=2,
            )
        )
        logger.info("Wrote benchmark JSON to %s", path)


def ensure_tensorrt_loadable() -> bool:
    """Make the pip-installed TensorRT libraries visible to ONNX Runtime.

    ``libonnxruntime_providers_tensorrt.so`` has three unresolved
    dependencies when TensorRT is installed from wheels, and they live in two
    different site-packages directories:

    * ``libnvinfer.so.10`` and ``libnvonnxparser.so.10`` in ``tensorrt_libs/``
    * ``libcudnn.so.9`` in ``nvidia/cudnn/lib/``

    Neither is on the dynamic linker's search path. ONNX Runtime ``dlopen``s
    the provider, resolution fails, and ORT then **silently falls back to
    CPU** — the failure surfaces only as suspiciously slow "TensorRT" numbers.

    Setting ``LD_LIBRARY_PATH`` here would be too late: the linker reads it at
    process start. Instead each dependency is loaded explicitly with
    ``RTLD_GLOBAL``, placing its symbols in the global namespace so the
    provider's later ``dlopen`` resolves against the already-loaded copies.

    Returns whether every required library was loaded.
    """
    import ctypes
    import importlib.util

    def package_dir(name: str) -> Path | None:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            return None
        return Path(spec.origin).parent

    search_dirs: list[Path] = []
    if (trt_dir := package_dir("tensorrt_libs")) is not None:
        search_dirs.append(trt_dir)
    if (nvidia_dir := package_dir("nvidia")) is not None:
        search_dirs.append(nvidia_dir / "cudnn" / "lib")

    if not search_dirs:
        logger.info("TensorRT wheels not installed; TensorRT benchmarks will be skipped")
        return False

    # cuDNN first, then nvinfer, then the parser that depends on nvinfer.
    required = ("libcudnn.so.9", "libnvinfer.so.10", "libnvonnxparser.so.10")

    for name in required:
        path = next((d / name for d in search_dirs if (d / name).exists()), None)
        if path is None:
            logger.warning(
                "TensorRT dependency %s not found under %s; skipping TensorRT",
                name,
                ", ".join(str(d) for d in search_dirs),
            )
            return False
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            logger.warning("Failed to preload %s: %s", path, exc)
            return False

    logger.info("Preloaded TensorRT dependencies from %s", search_dirs[0])
    return True


def capture_environment() -> dict[str, Any]:
    """Record hardware and library versions for reproducibility."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
            major, minor = torch.cuda.get_device_capability(0)
            env["compute_capability"] = f"sm_{major}{minor}"
            env["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except ImportError:
        pass

    try:
        import onnxruntime as ort

        env["onnxruntime"] = ort.__version__
        env["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        pass

    try:
        import tensorrt

        env["tensorrt"] = tensorrt.__version__
    except ImportError:
        env["tensorrt"] = None

    return env


def _summarise(
    timings_ms: list[float],
    *,
    backend: str,
    precision: str,
    batch_size: int,
    device: str,
    warmup: int,
    **extra: Any,
) -> LatencyResult:
    """Reduce raw per-iteration timings to a :class:`LatencyResult`."""
    ordered = sorted(timings_ms)

    def percentile(p: float) -> float:
        # Nearest-rank; exact enough at these sample counts and avoids
        # interpolating between measurements that were never observed.
        index = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[index]

    mean_ms = statistics.fmean(timings_ms)
    return LatencyResult(
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        device=device,
        mean_ms=mean_ms,
        std_ms=statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        p50_ms=percentile(0.50),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        min_ms=ordered[0],
        max_ms=ordered[-1],
        throughput_ips=batch_size * 1000.0 / mean_ms,
        iterations=len(timings_ms),
        warmup=warmup,
        **extra,
    )


def _time_callable(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    synchronise: Callable[[], None] | None = None,
) -> list[float]:
    """Time ``fn`` over ``iterations`` runs after ``warmup`` discarded runs."""
    for _ in range(warmup):
        fn()
    if synchronise:
        synchronise()

    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if synchronise:
            synchronise()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def benchmark_torch(
    model: Any,
    *,
    batch_size: int = 1,
    image_size: int = 224,
    device: str = "cuda",
    precision: str = "fp32",
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    compile_model: bool = False,
) -> LatencyResult:
    """Benchmark a PyTorch model.

    ``torch.compile`` warmup is deliberately longer: the first call triggers
    graph capture and kernel autotuning, which can take tens of seconds and
    would otherwise dominate the measurement.
    """
    import torch

    model = model.eval().to(device)
    if compile_model:
        model = torch.compile(model)
        warmup = max(warmup, 30)

    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    def run() -> None:
        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype):
                    model(inputs)
            else:
                model(inputs)

    synchronise = torch.cuda.synchronize if device.startswith("cuda") else None
    timings = _time_callable(run, warmup=warmup, iterations=iterations, synchronise=synchronise)

    # The device belongs in the backend label. Without it, CPU and CUDA rows are
    # indistinguishable in the report and a baseline lookup keyed on the label
    # silently matches the wrong one.
    device_tag = "cuda" if device.startswith("cuda") else "cpu"
    backend = f"pytorch-{device_tag}-compiled" if compile_model else f"pytorch-{device_tag}"
    return _summarise(
        timings,
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        device=device,
        warmup=warmup,
    )


def benchmark_onnx(
    model_path: Path,
    *,
    batch_size: int = 1,
    image_size: int = 224,
    provider: str = "CPUExecutionProvider",
    precision: str = "fp32",
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    provider_options: dict[str, Any] | None = None,
    trt_cache_dir: Path | None = None,
) -> LatencyResult:
    """Benchmark an ONNX model under a given execution provider.

    TensorRT builds an optimised engine on first use, which takes minutes. The
    engine cache is enabled and pointed at ``trt_cache_dir`` so repeated runs
    reuse it; without caching, TensorRT results would mostly measure engine
    construction.
    """
    import onnxruntime as ort

    options = dict(provider_options or {})
    if provider == "TensorrtExecutionProvider":
        cache = trt_cache_dir or model_path.parent / "trt_cache"
        cache.mkdir(parents=True, exist_ok=True)
        options.setdefault("trt_engine_cache_enable", True)
        options.setdefault("trt_engine_cache_path", str(cache))
        options.setdefault("trt_fp16_enable", precision == "fp16")
        options.setdefault("trt_int8_enable", precision == "int8")
        # Engine building is slow; a longer warmup absorbs it.
        warmup = max(warmup, 5)

    # Provider options (the TensorRT engine cache) cannot go through the shared
    # helper's simple provider list, so the session is built here -- but the
    # preload and the provider assertion are still applied, because measuring
    # the wrong backend is exactly the failure this harness exists to avoid.
    from api.services.runtime import preload_cuda_libraries, preload_tensorrt_libraries

    if provider == "TensorrtExecutionProvider":
        preload_tensorrt_libraries()
    elif provider != "CPUExecutionProvider":
        preload_cuda_libraries()

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers: list[Any] = [(provider, options)] if options else [provider]
    # ORT requires a CPU fallback in the provider list.
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(
        str(model_path), sess_options=session_options, providers=providers
    )

    actual = session.get_providers()[0]
    if actual != provider:
        raise RuntimeError(
            f"Requested provider {provider} but ONNX Runtime selected {actual}. "
            f"Available: {session.get_providers()}. Benchmarking would silently "
            f"measure the wrong backend."
        )

    input_name = session.get_inputs()[0].name
    inputs = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)

    def run() -> None:
        session.run(None, {input_name: inputs})

    timings = _time_callable(run, warmup=warmup, iterations=iterations)

    device = "cpu" if provider == "CPUExecutionProvider" else "cuda"
    backend = {
        "CPUExecutionProvider": "onnxruntime-cpu",
        "CUDAExecutionProvider": "onnxruntime-cuda",
        "TensorrtExecutionProvider": "onnxruntime-tensorrt",
    }.get(provider, provider)

    return _summarise(
        timings,
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        device=device,
        warmup=warmup,
        model_size_mb=round(model_path.stat().st_size / 1e6, 1),
    )


def format_markdown_report(
    report: BenchmarkReport,
    *,
    baseline_backend: str = "pytorch-cuda",
    baseline_precision: str = "fp32",
) -> str:
    """Render a benchmark report as a Markdown document.

    Speedups are expressed relative to eager PyTorch at the same batch size,
    since an absolute millisecond figure means little without a reference
    point. The baseline is matched on backend *and* precision: matching on
    backend alone would let a bf16 or compiled row overwrite the fp32 entry and
    silently rebase every speedup in the table.

    Falls back to CPU when the machine has no GPU, so the report is still
    meaningful on a CPU-only host.
    """
    lines: list[str] = [
        "# Inference Performance Benchmarks",
        "",
        f"Generated: {report.generated_at}",
        "",
        "## Environment",
        "",
        "| Property | Value |",
        "| --- | --- |",
    ]

    for key, value in report.environment.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        lines.append(f"| {key} | {value} |")

    available = {r.backend for r in report.results}
    if baseline_backend not in available:
        baseline_backend = "pytorch-cpu"

    # Matched on backend *and* precision. Matching on backend alone lets the
    # bf16 or compiled row overwrite the fp32 entry, silently rebasing every
    # speedup in the table against whichever configuration happened to run last.
    # Keyed by (model, batch): a speedup comparing a detector against a
    # classifier baseline would be meaningless.
    baselines = {
        (r.model, r.batch_size): r.mean_ms
        for r in report.results
        if r.backend == baseline_backend and r.precision == baseline_precision
    }

    by_model: dict[str, list[LatencyResult]] = {}
    for result in report.results:
        by_model.setdefault(result.model, []).append(result)

    lines += [
        "",
        "## Latency",
        "",
        f"All timings exclude warmup and synchronise the device before stopping "
        f"the clock. Speedup is relative to `{baseline_backend}` "
        f"({baseline_precision}) at the same batch size.",
        "",
        "| Model | Backend | Precision | Batch | Mean (ms) | p50 | p95 | p99 | Std | "
        "Throughput (img/s) | Per-image (ms) | Speedup | Size (MB) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in sorted(
        report.results, key=lambda x: (x.model != "classifier", x.model, x.batch_size, x.mean_ms)
    ):
        base = baselines.get((r.model, r.batch_size))
        speedup = f"{base / r.mean_ms:.2f}x" if base else "-"
        size = f"{r.model_size_mb:.1f}" if r.model_size_mb is not None else "-"
        lines.append(
            f"| {r.model} | {r.backend} | {r.precision} | {r.batch_size} | {r.mean_ms:.2f} | "
            f"{r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.p99_ms:.2f} | {r.std_ms:.2f} | "
            f"{r.throughput_ips:.1f} | {r.per_image_ms:.2f} | {speedup} | {size} |"
        )

    accuracy_rows = [r for r in report.results if r.accuracy_top1 is not None]
    if accuracy_rows:
        lines += [
            "",
            "## Accuracy",
            "",
            "| Backend | Precision | Top-1 | Notes |",
            "| --- | --- | ---: | --- |",
        ]
        for r in accuracy_rows:
            # Narrowed by the accuracy_rows filter above; mypy cannot see it.
            top1 = r.accuracy_top1 or 0.0
            lines.append(f"| {r.backend} | {r.precision} | {top1 * 100:.2f}% | {r.notes} |")

    return "\n".join(lines) + "\n"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
