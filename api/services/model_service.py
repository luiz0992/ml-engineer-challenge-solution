"""Model loading, versioning, and backend selection.

Responsible for three things the challenge calls out explicitly:

* **Model versioning.** Multiple versions of a model can be registered
  simultaneously; one is active and serves unpinned requests, while clients may
  pin a specific version. This is what makes a canary or rollback possible
  without redeploying.
* **Graceful degradation.** Backends are tried in preference order. If
  TensorRT cannot initialise — a missing library, an unbuildable engine, a GPU
  that disappeared — the service falls back to CUDA, then to CPU, rather than
  failing to start. A degraded service that answers is better than a healthy
  one that does not exist, provided the degradation is *visible*, so every
  fallback is logged and surfaced in ``/health`` and the model metadata.
* **Provenance.** Every loaded model reports the artefact and backend actually
  in use, so a prediction can be attributed after the fact.

Sessions are created once at startup and reused. ONNX Runtime sessions are
thread-safe for inference and expensive to build, so per-request construction
would dominate latency.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from api.config import InferenceBackend, Settings
from api.exceptions import ModelNotFoundError, ModelUnavailableError
from api.logging_config import get_logger
from api.models.schemas import TaskType
from api.services.runtime import preload_cuda_libraries, preload_tensorrt_libraries

logger = get_logger(__name__)

#: Fallback order per preferred backend, most to least capable. Every chain
#: ends at CPU, which has no external dependencies and always works.
_FALLBACK_CHAINS: dict[InferenceBackend, tuple[InferenceBackend, ...]] = {
    InferenceBackend.TENSORRT: (
        InferenceBackend.TENSORRT,
        InferenceBackend.ONNX,
        InferenceBackend.ONNX_INT8,
    ),
    InferenceBackend.ONNX: (InferenceBackend.ONNX, InferenceBackend.ONNX_INT8),
    InferenceBackend.ONNX_INT8: (InferenceBackend.ONNX_INT8, InferenceBackend.ONNX),
    InferenceBackend.TORCH: (InferenceBackend.TORCH, InferenceBackend.ONNX),
}

_PROVIDER_FOR_BACKEND: dict[InferenceBackend, list[str]] = {
    InferenceBackend.TENSORRT: ["TensorrtExecutionProvider", "CPUExecutionProvider"],
    InferenceBackend.ONNX: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    InferenceBackend.ONNX_INT8: ["CPUExecutionProvider"],
}


@dataclass(slots=True)
class LoadedModel:
    """A model that is resident and ready to serve."""

    name: str
    version: str
    task: TaskType
    backend: InferenceBackend
    session: Any
    input_name: str
    class_names: list[str]
    wnids: list[str]
    image_size: int
    artifact_path: Path
    loaded_at: datetime
    metrics: dict[str, float] = field(default_factory=dict)
    #: Set when the requested backend could not be used.
    degraded_from: InferenceBackend | None = None

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def artifact_size_mb(self) -> float:
        return round(self.artifact_path.stat().st_size / 1e6, 1)


class ModelService:
    """Registry of loaded models, with versioning and backend fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._models: dict[str, LoadedModel] = {}
        self._active: dict[str, str] = {}  # model name -> active version
        # Guards mutation of the registry. Inference itself needs no lock, as
        # ONNX Runtime sessions are thread-safe for Run().
        self._lock = threading.RLock()

    # --- Loading ----------------------------------------------------------
    def load_classifier(
        self,
        *,
        name: str = "tiny-imagenet-classifier",
        version: str = "v1",
        make_active: bool = True,
    ) -> LoadedModel:
        """Load the classification model, degrading through the backend chain.

        Raises :class:`ModelUnavailableError` only if *every* backend fails,
        which means the artefacts are missing or corrupt rather than that one
        runtime is unavailable.
        """
        artifacts = self.settings.artifacts_dir
        metadata = self._read_classifier_metadata(artifacts)

        preferred = self.settings.inference_backend
        chain = (
            _FALLBACK_CHAINS.get(preferred, (preferred,))
            if self.settings.enable_graceful_degradation
            else (preferred,)
        )

        failures: list[str] = []
        for candidate in chain:
            artifact = self._artifact_for(artifacts, candidate)
            if artifact is None or not artifact.exists():
                failures.append(f"{candidate.value}: artefact not found")
                continue
            try:
                session, input_name = self._create_session(artifact, candidate)
                # Prove the backend actually executes before accepting it.
                self._warmup(session, input_name, metadata["image_size"], candidate)
            except Exception as exc:
                failures.append(f"{candidate.value}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "backend_unavailable",
                    backend=candidate.value,
                    artifact=str(artifact),
                    error=str(exc),
                )
                continue

            model = LoadedModel(
                name=name,
                version=version,
                task=TaskType.CLASSIFICATION,
                backend=candidate,
                session=session,
                input_name=input_name,
                class_names=metadata["class_names"],
                wnids=metadata["wnids"],
                image_size=metadata["image_size"],
                artifact_path=artifact,
                loaded_at=datetime.now(UTC),
                metrics=metadata.get("metrics", {}),
                degraded_from=preferred if candidate is not preferred else None,
            )
            self.register(model, make_active=make_active)

            if model.degraded_from is not None:
                # Loud: the service is running, but not as configured.
                logger.warning(
                    "model_loaded_degraded",
                    model=model.key,
                    requested_backend=preferred.value,
                    actual_backend=candidate.value,
                    reason="; ".join(failures),
                )
            else:
                logger.info(
                    "model_loaded",
                    model=model.key,
                    backend=candidate.value,
                    classes=model.num_classes,
                    artifact_mb=model.artifact_size_mb,
                )
            return model

        raise ModelUnavailableError(
            "No inference backend could be initialised for the classifier.",
            details={"attempts": failures},
        )

    def _read_classifier_metadata(self, artifacts_dir: Path) -> dict[str, Any]:
        """Load label and input metadata exported alongside the model.

        Labels come from the training run rather than being re-derived from the
        dataset: the serving host may not have the dataset at all, and any
        difference in directory ordering would silently permute every label.
        """
        labels_path = artifacts_dir / "labels.json"
        if not labels_path.exists():
            raise ModelUnavailableError(
                f"Model labels not found at {labels_path}. Export artefacts with "
                f"`scripts/prepare_artifacts.py` before starting the API.",
                details={"expected_path": str(labels_path)},
            )

        labels = json.loads(labels_path.read_text())
        metadata: dict[str, Any] = {
            "class_names": labels["class_names"],
            "wnids": labels.get("wnids", []),
            "image_size": labels.get("image_size", 224),
            "metrics": {},
        }

        metrics_path = artifacts_dir / "metrics.json"
        if metrics_path.exists():
            metadata["metrics"] = {
                k: float(v)
                for k, v in json.loads(metrics_path.read_text()).items()
                if isinstance(v, int | float)
            }
        return metadata

    @staticmethod
    def _artifact_for(artifacts_dir: Path, backend: InferenceBackend) -> Path | None:
        """Map a backend to the artefact file it consumes."""
        onnx_dir = artifacts_dir / "onnx"
        return {
            # TensorRT and CUDA consume the same FP32 graph; the provider, not
            # the file, is what differs.
            InferenceBackend.TENSORRT: onnx_dir / "classifier_fp32.onnx",
            InferenceBackend.ONNX: onnx_dir / "classifier_fp32.onnx",
            InferenceBackend.ONNX_INT8: onnx_dir / "classifier_int8.onnx",
        }.get(backend)

    def _create_session(self, artifact: Path, backend: InferenceBackend) -> tuple[Any, str]:
        """Create an ONNX Runtime session, verifying the provider took effect.

        ONNX Runtime silently falls back to CPU when a requested provider
        cannot load. Without this check the service would report itself as
        running TensorRT while executing on CPU roughly a hundred times slower.
        """
        import onnxruntime as ort

        # Preload GPU libraries before session creation; see api.services.runtime.
        if backend is InferenceBackend.TENSORRT:
            preload_tensorrt_libraries()
        elif backend is InferenceBackend.ONNX:
            preload_cuda_libraries()

        providers = _PROVIDER_FOR_BACKEND.get(backend, ["CPUExecutionProvider"])
        available = set(ort.get_available_providers())
        requested = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(str(artifact), sess_options=options, providers=requested)

        selected = session.get_providers()[0]
        if selected != requested[0]:
            raise RuntimeError(
                f"Requested provider {requested[0]} but ONNX Runtime selected "
                f"{selected}; refusing to report a backend that is not in use."
            )

        return session, session.get_inputs()[0].name

    @staticmethod
    def _warmup(
        session: Any,
        input_name: str,
        image_size: int,
        backend: InferenceBackend,
        output_ndim: int = 2,
    ) -> None:
        """Run one throwaway inference to prove the backend works.

        This is not an optimisation, it is a correctness check. A successfully
        created session does not imply a working backend: the CUDA provider
        resolves cuDNN lazily, so a session builds cleanly and then fails on
        every request with NOT_IMPLEMENTED once a kernel runs. Without this
        probe the service would start, pass its provider check, report itself
        healthy, and return 503 for all traffic.

        Executing here means such a backend is rejected during loading and the
        fallback chain moves to the next candidate, which is what makes
        graceful degradation real rather than aspirational.

        It also pays the one-off initialisation cost (context creation, kernel
        autotuning, TensorRT engine deserialisation) at startup rather than in
        the first user request.
        """
        probe = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
        started = time.perf_counter()
        outputs = session.run(None, {input_name: probe})

        if not outputs or outputs[0].ndim != output_ndim:
            raise RuntimeError(
                f"Warmup produced an unexpected output shape for {backend.value}: "
                f"{[o.shape for o in outputs]}"
            )

        logger.debug(
            "backend_warmup_ok",
            backend=backend.value,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            output_shape=list(outputs[0].shape),
        )

    # --- Registry ---------------------------------------------------------
    def register(self, model: LoadedModel, *, make_active: bool = True) -> None:
        with self._lock:
            self._models[model.key] = model
            if make_active or model.name not in self._active:
                self._active[model.name] = model.version

    def get(self, name: str, version: str | None = None) -> LoadedModel:
        """Resolve a model, defaulting to the active version."""
        with self._lock:
            resolved = version or self._active.get(name)
            if resolved is None:
                raise ModelNotFoundError(
                    f"No model registered under {name!r}.",
                    details={"available": sorted(self._active)},
                )

            model = self._models.get(f"{name}:{resolved}")
            if model is None:
                raise ModelNotFoundError(
                    f"Model {name!r} version {resolved!r} is not registered.",
                    details={
                        "requested_version": resolved,
                        "available_versions": sorted(
                            m.version for m in self._models.values() if m.name == name
                        ),
                    },
                )
            return model

    def get_classifier(self, version: str | None = None) -> LoadedModel:
        return self.get("tiny-imagenet-classifier", version)

    def get_detector(self, version: str | None = None) -> LoadedModel:
        return self.get("rtdetr-coco-detector", version)

    def load_detector(
        self,
        *,
        name: str = "rtdetr-coco-detector",
        version: str = "v1",
        make_active: bool = True,
    ) -> LoadedModel:
        """Load the object detector.

        Unlike the classifier, a missing detector is not fatal. The service is
        useful without it, so the caller decides whether to treat failure as an
        error; ``/api/v1/detect`` reports 503 with an explanation when the model
        is absent.
        """
        artifacts = self.settings.artifacts_dir
        artifact = artifacts / "onnx" / "detector_fp32.onnx"
        labels_path = artifacts / "detection_labels.json"

        if not artifact.exists() or not labels_path.exists():
            raise ModelUnavailableError(
                "Detection artefacts not found. Export them with "
                "`python scripts/prepare_artifacts.py --with-detection`.",
                details={"expected": [str(artifact), str(labels_path)]},
            )

        metadata = json.loads(labels_path.read_text())
        id2label = {int(k): v for k, v in metadata["id2label"].items()}
        class_names = [id2label.get(i, str(i)) for i in range(metadata["num_classes"])]

        # Detection runs on the same ONNX backends as the classifier; the
        # fallback chain is shared.
        preferred = self.settings.inference_backend
        chain = (
            _FALLBACK_CHAINS.get(preferred, (preferred,))
            if self.settings.enable_graceful_degradation
            else (preferred,)
        )
        # INT8 is not exported for the detector, so it is not a candidate.
        chain = tuple(c for c in chain if c is not InferenceBackend.ONNX_INT8)

        failures: list[str] = []
        for candidate in chain:
            try:
                session, input_name = self._create_session(artifact, candidate)
                self._warmup(session, input_name, metadata["image_size"], candidate, output_ndim=3)
            except Exception as exc:
                failures.append(f"{candidate.value}: {type(exc).__name__}: {exc}")
                continue

            model = LoadedModel(
                name=name,
                version=version,
                task=TaskType.DETECTION,
                backend=candidate,
                session=session,
                input_name=input_name,
                class_names=class_names,
                wnids=[],
                image_size=metadata["image_size"],
                artifact_path=artifact,
                loaded_at=datetime.now(UTC),
                metrics={},
                degraded_from=preferred if candidate is not preferred else None,
            )
            self.register(model, make_active=make_active)
            logger.info(
                "model_loaded",
                model=model.key,
                backend=candidate.value,
                classes=model.num_classes,
                task="detection",
            )
            return model

        raise ModelUnavailableError(
            "No inference backend could be initialised for the detector.",
            details={"attempts": failures},
        )

    def list_models(self) -> list[LoadedModel]:
        with self._lock:
            return list(self._models.values())

    def is_active(self, model: LoadedModel) -> bool:
        with self._lock:
            return self._active.get(model.name) == model.version

    def set_active(self, name: str, version: str) -> None:
        """Promote a version to serve unpinned traffic.

        The target must already be loaded, so promotion cannot point traffic at
        an artefact that has never been successfully initialised.
        """
        with self._lock:
            if f"{name}:{version}" not in self._models:
                raise ModelNotFoundError(f"Cannot activate {name}:{version}; it is not loaded.")
            previous = self._active.get(name)
            self._active[name] = version
        logger.info("model_activated", model=name, version=version, previous=previous)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return bool(self._models)

    # --- Inference --------------------------------------------------------
    def run(self, model: LoadedModel, batch: np.ndarray) -> np.ndarray:
        """Execute a forward pass, returning raw logits."""
        started = time.perf_counter()
        try:
            outputs = model.session.run(None, {model.input_name: batch})
        except Exception as exc:
            logger.exception("inference_failed", model=model.key, batch=batch.shape[0])
            raise ModelUnavailableError("The model failed to execute this request.") from exc

        logger.debug(
            "inference_complete",
            model=model.key,
            batch=batch.shape[0],
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return outputs[0]

    def run_multi(self, model: LoadedModel, batch: np.ndarray) -> list[np.ndarray]:
        """Execute a forward pass returning every output.

        Detection produces two tensors (scores and boxes); :meth:`run` returns
        only the first, which is the right shape for classification.
        """
        try:
            return model.session.run(None, {model.input_name: batch})
        except Exception as exc:
            logger.exception("inference_failed", model=model.key, batch=batch.shape[0])
            raise ModelUnavailableError("The model failed to execute this request.") from exc
