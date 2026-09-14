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
