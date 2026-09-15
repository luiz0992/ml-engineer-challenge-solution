"""GPU runtime library discovery.

NVIDIA's pip wheels install shared objects into per-package directories under
``site-packages`` that the dynamic linker does not search:

* ``nvidia/cudnn/lib/`` — ``libcudnn.so``
* ``nvidia/cublas/lib/``, ``nvidia/cufft/lib/`` and friends
* ``tensorrt_libs/`` — ``libnvinfer.so``, ``libnvonnxparser.so``

ONNX Runtime ``dlopen``s these lazily. When resolution fails the consequences
differ by provider and are both bad:

* **TensorRT** fails at session creation and ORT falls back to CPU silently.
* **CUDA** creates the session successfully and then fails at *inference*,
  because cuDNN is only needed once a kernel runs. A service can therefore
  start, report itself healthy, pass a provider check, and fail every single
  request.

Setting ``LD_LIBRARY_PATH`` from inside the process does not help: the linker
reads it at exec time. The libraries are instead loaded explicitly with
``RTLD_GLOBAL``, which puts their symbols in the global namespace so ORT's
later ``dlopen`` resolves against the already-loaded copies.

This module lives under ``api/`` rather than ``models/`` so the serving image
does not need the training package.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

from api.logging_config import get_logger

logger = get_logger(__name__)

#: Loaded in dependency order: cuDNN needs cuBLAS, the ONNX parser needs
#: nvinfer. Loading a dependant first leaves unresolved symbols.
_CUDA_LIBRARIES: tuple[tuple[str, str], ...] = (
    ("nvidia/cublas/lib", "libcublas.so.12"),
    ("nvidia/cublas/lib", "libcublasLt.so.12"),
    ("nvidia/cudnn/lib", "libcudnn.so.9"),
    # ORT dlopens the unversioned name; the wheel ships only the versioned one.
    ("nvidia/cudnn/lib", "libcudnn.so"),
)

_TENSORRT_LIBRARIES: tuple[tuple[str, str], ...] = (
    ("tensorrt_libs", "libnvinfer.so.10"),
    ("tensorrt_libs", "libnvonnxparser.so.10"),
)


def _site_packages() -> Path | None:
    """Locate the site-packages directory holding the NVIDIA wheels."""
    spec = importlib.util.find_spec("numpy")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent.parent


def _load(paths: tuple[tuple[str, str], ...], *, label: str) -> bool:
    """Preload a set of libraries. Returns whether all of them loaded."""
    root = _site_packages()
    if root is None:
        return False

    loaded = 0
    for subdir, name in paths:
        candidate = root / subdir / name
        if not candidate.exists():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            loaded += 1
        except OSError as exc:
            logger.debug("library_preload_failed", library=name, error=str(exc))

    if loaded:
        logger.info("gpu_libraries_preloaded", runtime=label, count=loaded)
    return loaded > 0


def preload_cuda_libraries() -> bool:
    """Make cuDNN and cuBLAS resolvable for the CUDA execution provider."""
    return _load(_CUDA_LIBRARIES, label="cuda")


def preload_tensorrt_libraries() -> bool:
    """Make TensorRT resolvable for the TensorRT execution provider.

    TensorRT also needs the CUDA libraries, so those are loaded first.
    """
    preload_cuda_libraries()
    return _load(_TENSORRT_LIBRARIES, label="tensorrt")


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------
#: Guards the one-time preload. ctypes.CDLL is idempotent, but the work and the
#: log line are not worth repeating per session.
_PRELOADED = False

#: cgroup files describing the CPU quota, newest interface first.
_CGROUP_V2_QUOTA = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def available_cpus() -> int:
    """Return the CPUs this process may actually use.

    `os.cpu_count()` reports the *host's* processors, and a container is not
    told about its own cgroup quota. ONNX Runtime sizes its intra-op thread
    pool from that number, so a 4-CPU container on a 32-core host builds a
    32-thread pool and then thrashes inside a quota a quarter that size.

    Measured on this project's own image, classifying one image on CPU:

    ========================  =========
    intra_op_num_threads      p50
    ========================  =========
    default (32, host count)  102.8 ms
    4 (matching the quota)     12.4 ms
    ========================  =========

    An 8x difference from one number, and silent -- the container is healthy,
    correct, and slow. CPU affinity is checked as well as the quota, because a
    pinned process is limited by affinity even with no quota set.
    """
    limit = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    limit = limit or 1

    quota = _cgroup_cpu_quota()
    if quota is not None:
        limit = min(limit, quota)

    return max(1, limit)


def _cgroup_cpu_quota() -> int | None:
    """Read the CPU quota in whole CPUs, or ``None`` when unlimited."""
    try:
        if _CGROUP_V2_QUOTA.exists():
            # "<quota> <period>", or "max <period>" when uncapped.
            raw_quota, raw_period = _CGROUP_V2_QUOTA.read_text().split()
            if raw_quota == "max":
                return None
            return max(1, int(float(raw_quota) / float(raw_period)))

        if _CGROUP_V1_QUOTA.exists() and _CGROUP_V1_PERIOD.exists():
            # A negative quota means uncapped under cgroup v1.
            micros = int(_CGROUP_V1_QUOTA.read_text())
            if micros <= 0:
                return None
            return max(1, micros // int(_CGROUP_V1_PERIOD.read_text()))
    except (OSError, ValueError) as exc:
        # Never fatal: an unreadable cgroup file means we fall back to the
        # affinity count, which is worse but not wrong.
        logger.debug("cgroup_cpu_quota_unreadable", error=str(exc))

    return None


def create_session(
    model_path: str,
    *,
    providers: list[str] | None = None,
    satisfied_by: Collection[str] | None = None,
    graph_optimization: bool = True,
    verify_provider: bool = True,
) -> tuple[Any, str]:
    """Create an ONNX Runtime session with GPU libraries already resolved.

    **Every** session in this project must be created through this function
    rather than by calling ``ort.InferenceSession`` directly, because the
    failure it prevents is silent in two different ways:

    * **TensorRT** fails at session creation and ONNX Runtime falls back to CPU
      without raising. The session works; it is simply about a hundred times
      slower than the caller believes.
    * **CUDA** resolves cuDNN lazily at the first kernel launch. The session is
      created successfully, reports the provider it was asked for, and then
      fails *every* inference with ``NOT_IMPLEMENTED``.

    That second failure recurred three times during development -- in the model
    service, in the benchmark harness, and in the evaluation scripts -- because
    each call site had to remember to preload. A shared constructor removes the
    opportunity to forget. :func:`~api.services.model_service.ModelService._warmup`
    remains the backstop that proves a backend actually executes.

    ``satisfied_by`` names the providers that legitimately fulfil the caller's
    request, which is not the same as the providers handed to ONNX Runtime.
    ONNX Runtime requires a CPU entry in every provider list, so CPU appears in
    the TensorRT list as a mechanical requirement rather than an acceptable
    outcome -- a session that lands on CPU has not delivered TensorRT. For a
    plain ONNX session the opposite holds: CPU is exactly what a CPU-only
    deployment was built for, and calling that a degradation would make the
    degraded-backend alert permanently red on the default configuration.

    Defaults to ``{providers[0]}``, the strict reading, so a caller that asks
    for one specific accelerator gets one.

    Returns ``(session, input_name)``.
    """
    global _PRELOADED

    import onnxruntime as ort

    requested = providers or ["CPUExecutionProvider"]

    if not _PRELOADED and any(p != "CPUExecutionProvider" for p in requested):
        if any("Tensorrt" in p for p in requested):
            preload_tensorrt_libraries()
        else:
            preload_cuda_libraries()
        _PRELOADED = True

    available = set(ort.get_available_providers())
    # The provider the caller actually asked for, captured before filtering.
    # Verifying against the *filtered* list is what let this check pass
    # vacuously: if the requested provider is not registered in this build it
    # is filtered out, CPU is appended, and `usable[0]` becomes CPU -- so the
    # comparison below succeeded in precisely the case it exists to catch.
    preferred = requested[0]

    usable = [p for p in requested if p in available]
    if "CPUExecutionProvider" not in usable:
        # ONNX Runtime requires a CPU fallback in the provider list.
        usable.append("CPUExecutionProvider")

    options = ort.SessionOptions()
    if graph_optimization:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Sized from the cgroup quota rather than left to ONNX Runtime's default of
    # the host processor count. See `available_cpus` for the 8x this is worth
    # inside a CPU-limited container.
    #
    # Applied to every session, including GPU ones. An earlier version skipped
    # this whenever CUDA or TensorRT was selected, on the theory that the GPU
    # does the work -- but a GPU container still has a CPU quota, still runs
    # any operator the provider does not support on the CPU, and still pays for
    # a thread pool sized for hardware it cannot use. Measured on CUDA at batch
    # 8, the bound costs nothing: 3.45 ms at the host default of 32 threads
    # against 3.45 ms at 4.
    threads = available_cpus()
    options.intra_op_num_threads = threads
    # One inter-op thread: these graphs are a single sequential chain, so
    # parallelising across nodes only adds contention with the intra-op
    # pool that is doing the real work.
    options.inter_op_num_threads = 1
    logger.debug("onnx_threads_configured", intra_op=threads, model=model_path)

    session = ort.InferenceSession(model_path, sess_options=options, providers=usable)

    if verify_provider:
        acceptable = frozenset(satisfied_by) if satisfied_by is not None else frozenset({preferred})
        selected = session.get_providers()[0]

        if selected not in acceptable:
            # Raising rather than quietly proceeding is what lets ModelService's
            # fallback chain record `degraded_from` honestly. Returning the
            # session anyway would have the API advertise `backend=tensorrt` on
            # every response, health check and audit row while running two
            # orders of magnitude slower on CPU.
            raise RuntimeError(
                f"ONNX Runtime selected {selected}, which does not satisfy the "
                f"requested {sorted(acceptable)} "
                f"(available: {sorted(available)}). Refusing to report a "
                f"backend that is not in use."
            )

    return session, session.get_inputs()[0].name
