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


def create_session(
    model_path: str,
    *,
    providers: list[str] | None = None,
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
    usable = [p for p in requested if p in available]
    if "CPUExecutionProvider" not in usable:
        # ONNX Runtime requires a CPU fallback in the provider list.
        usable.append("CPUExecutionProvider")

    options = ort.SessionOptions()
    if graph_optimization:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=options, providers=usable)

    if verify_provider and usable[0] != "CPUExecutionProvider":
        selected = session.get_providers()[0]
        if selected != usable[0]:
            raise RuntimeError(
                f"Requested provider {usable[0]} but ONNX Runtime selected "
                f"{selected}. Refusing to report a backend that is not in use."
            )

    return session, session.get_inputs()[0].name
