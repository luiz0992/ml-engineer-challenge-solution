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
import re
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
from api.services.runtime import create_session

logger = get_logger(__name__)

#: The version served when a caller does not pin one, and the only version the
#: artefact pipeline produces by default.
DEFAULT_VERSION = "v1"

#: Version directory names must look like a version, not like a stray directory
#: the artefact build left behind. Digits and dots only, so `v1` and `v2.1.0`
#: are versions while `trt_cache` and a half-written `v2.tmp-partial` are not.
_VERSION_PATTERN = re.compile(r"^v\d+(?:\.\d+)*$")


def resolve_versioned_artifact(onnx_dir: Path, filename: str, version: str) -> Path:
    """Locate the artefact for one model version.

    Two layouts are supported, in this order:

    1. ``onnx/<version>/<filename>`` -- the versioned layout. Adding a second
       version is a matter of dropping a directory next to the first.
    2. ``onnx/<filename>`` -- the flat layout the pipeline produces today,
       accepted **only** for `DEFAULT_VERSION`.

    That restriction is the point. Falling back to the flat file for any
    version would mean a caller pinning `v2` silently receives `v1` weights
    while the response claims `v2` -- a version contract that lies, which is
    worse than a 404. Restricting the fallback makes a missing version fail
    loudly at load time instead.

    Returns the path whether or not it exists; the caller reports a missing
    artefact with the context it has.
    """
    versioned = onnx_dir / version / filename
    if versioned.exists() or version != DEFAULT_VERSION:
        return versioned
    return onnx_dir / filename


def resolve_versioned_metadata(artifacts_dir: Path, filename: str, version: str) -> Path | None:
    """Locate a version's label or metric sidecar, or ``None`` if it has none.

    Weights are versioned by `resolve_versioned_artifact`; the files that say
    what those weights *mean* have to be versioned by the same rule, or a `v2`
    built on a re-ordered or extended class list is served with `v1`'s label
    mapping. That mislabels every prediction, silently, while the response, the
    audit row and the A/B analysis all agree -- the same failure mode as
    serving the wrong weights, and harder to notice because the output still
    looks like a plausible class name.

    So a version directory must be self-describing: `onnx/<version>/` carries
    its own `labels.json` alongside its own graph. Only `DEFAULT_VERSION` falls
    back to the flat layout the artefact pipeline produces today.
    """
    versioned = artifacts_dir / "onnx" / version / filename
    if versioned.exists():
        return versioned

    if version == DEFAULT_VERSION:
        flat = artifacts_dir / filename
        if flat.exists():
            return flat

    return None


def discover_versions(onnx_dir: Path) -> list[str]:
    """List the versions present on disk, default first.

    Lets a deployment add a model version by shipping artefacts rather than by
    changing code, which is what makes the versioning support in the registry
    something more than a data structure.
    """
    found = {
        entry.name
        for entry in onnx_dir.glob("*")
        if entry.is_dir() and _VERSION_PATTERN.match(entry.name)
    }
    # Any graph at the top level means the flat layout is in use. Matching
    # only `*_fp32.onnx` missed a deployment shipping INT8 artefacts alone,
    # which `InferenceBackend.ONNX_INT8` makes a supported configuration.
    if any(onnx_dir.glob("*.onnx")):
        found.add(DEFAULT_VERSION)

    # Natural order, so v10 sorts after v2 rather than before it.
    def key(version: str) -> list[int]:
        return [int(part) for part in re.findall(r"\d+", version)]

    ordered = sorted(found - {DEFAULT_VERSION}, key=key)
    return ([DEFAULT_VERSION] if DEFAULT_VERSION in found else []) + ordered


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

#: Which providers actually *satisfy* each backend label, as distinct from the
#: list handed to ONNX Runtime above.
#:
#: The two differ because ONNX Runtime requires a CPU entry in every provider
#: list. CPU therefore appears alongside TensorRT as a mechanical requirement,
#: not as an acceptable outcome -- a session that lands on CPU has not
#: delivered TensorRT, and labelling it `tensorrt` would be a hundredfold lie.
#:
#: For plain `onnx` the opposite holds. CPU is what a CPU-only deployment was
#: built for, so treating it as a fallback would mark the default configuration
#: permanently degraded, fire `ModelRunningDegraded` forever, and push serving
#: down the chain to INT8 -- costing 5.4 points of top-1 accuracy to avoid a
#: degradation that was not one.
_SATISFYING_PROVIDERS: dict[InferenceBackend, frozenset[str]] = {
    InferenceBackend.TENSORRT: frozenset({"TensorrtExecutionProvider"}),
    InferenceBackend.ONNX: frozenset({"CUDAExecutionProvider", "CPUExecutionProvider"}),
    InferenceBackend.ONNX_INT8: frozenset({"CPUExecutionProvider"}),
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
        version: str = DEFAULT_VERSION,
        make_active: bool = True,
    ) -> LoadedModel:
        """Load the classification model, degrading through the backend chain.

        Raises :class:`ModelUnavailableError` only if *every* backend fails,
        which means the artefacts are missing or corrupt rather than that one
        runtime is unavailable.
        """
        artifacts = self.settings.artifacts_dir
        metadata = self._read_classifier_metadata(artifacts, version)

        preferred = self.settings.inference_backend
        chain = (
            _FALLBACK_CHAINS.get(preferred, (preferred,))
            if self.settings.enable_graceful_degradation
            else (preferred,)
        )

        failures: list[str] = []
        for candidate in chain:
            artifact = self._artifact_for(artifacts, candidate, version)
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

    def _read_classifier_metadata(
        self, artifacts_dir: Path, version: str = DEFAULT_VERSION
    ) -> dict[str, Any]:
        """Load label and input metadata exported alongside the model.

        Labels come from the training run rather than being re-derived from the
        dataset: the serving host may not have the dataset at all, and any
        difference in directory ordering would silently permute every label.
        """
        labels_path = resolve_versioned_metadata(artifacts_dir, "labels.json", version)
        if labels_path is None:
            if version == DEFAULT_VERSION:
                expected = artifacts_dir / "labels.json"
                detail = (
                    f"Model labels not found at {expected}. Export artefacts with "
                    f"`scripts/prepare_artifacts.py` before starting the API."
                )
            else:
                expected = artifacts_dir / "onnx" / version / "labels.json"
                detail = (
                    f"Model labels not found for version {version}. A non-default "
                    f"version must carry its own labels.json next to its graph, so "
                    f"its predictions cannot be named from another version's class "
                    f"list. Expected {expected}. Copy it from the run that produced "
                    f"the weights, or re-run `scripts/prepare_artifacts.py`."
                )
            raise ModelUnavailableError(
                detail, details={"expected_path": str(expected), "version": version}
            )

        labels = json.loads(labels_path.read_text())
        metadata: dict[str, Any] = {
            "class_names": labels["class_names"],
            "wnids": labels.get("wnids", []),
            "image_size": labels.get("image_size", 224),
            "metrics": {},
        }

        metrics_path = resolve_versioned_metadata(artifacts_dir, "metrics.json", version)
        if metrics_path is not None:
            metadata["metrics"] = {
                k: float(v)
                for k, v in json.loads(metrics_path.read_text()).items()
                if isinstance(v, int | float)
            }
        return metadata

    @staticmethod
    def _artifact_for(
        artifacts_dir: Path, backend: InferenceBackend, version: str = DEFAULT_VERSION
    ) -> Path | None:
        """Map a backend and version to the artefact file it consumes."""
        filename = {
            # TensorRT and CUDA consume the same FP32 graph; the provider, not
            # the file, is what differs.
            InferenceBackend.TENSORRT: "classifier_fp32.onnx",
            InferenceBackend.ONNX: "classifier_fp32.onnx",
            InferenceBackend.ONNX_INT8: "classifier_int8.onnx",
        }.get(backend)

        if filename is None:
            return None
        return resolve_versioned_artifact(artifacts_dir / "onnx", filename, version)

    def _create_session(self, artifact: Path, backend: InferenceBackend) -> tuple[Any, str]:
        """Create an ONNX Runtime session for ``backend``.

        Delegates to :func:`api.services.runtime.create_session`, which
        preloads GPU libraries and verifies the provider actually took effect.
        Calling ``ort.InferenceSession`` directly is what allowed the cuDNN
        lazy-resolution failure to recur in three separate places.
        """
        return create_session(
            str(artifact),
            providers=_PROVIDER_FOR_BACKEND.get(backend, ["CPUExecutionProvider"]),
            satisfied_by=_SATISFYING_PROVIDERS.get(backend, frozenset({"CPUExecutionProvider"})),
        )

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

    def get_embedder(self, version: str | None = None) -> LoadedModel:
        return self.get("tiny-imagenet-embedder", version)

    def get_for_backend(
        self, name: str, version: str | None, backend: InferenceBackend | None
    ) -> LoadedModel:
        """Resolve a model, optionally requiring a specific runtime.

        Used when a caller pins ``?backend=onnx-int8``. The default path
        still serves FP32; INT8 is available, measured, and opt-in, because
        forcing it on every request would silently spend the accuracy the
        calibration study documented.
        """
        model = self.get(name, version)
        if backend is None or model.backend is backend:
            return model
        raise ModelNotFoundError(
            f"Model {name!r} version {model.version!r} is loaded as "
            f"{model.backend.value}, not {backend.value}. Set "
            f"INFERENCE_BACKEND={backend.value} to serve that runtime.",
            details={
                "requested_backend": backend.value,
                "loaded_backend": model.backend.value,
            },
        )

    def load_embedder(
        self,
        *,
        name: str = "tiny-imagenet-embedder",
        version: str = DEFAULT_VERSION,
        make_active: bool = True,
    ) -> LoadedModel:
        """Load the embedding model backing similarity search.

        Shares the Tiny-ImageNet label list so an indexed image's class is
        named from the same mapping as classification.
        """
        artifacts = self.settings.artifacts_dir
        metadata = self._read_classifier_metadata(artifacts, version)

        preferred = self.settings.inference_backend
        chain = (
            _FALLBACK_CHAINS.get(preferred, (preferred,))
            if self.settings.enable_graceful_degradation
            else (preferred,)
        )
        # INT8 is served only when it is the preferred backend. Falling back
        # to it from FP32 would spend the 50% recall drop the calibration
        # study measured, silently.
        if preferred is not InferenceBackend.ONNX_INT8:
            chain = tuple(c for c in chain if c is not InferenceBackend.ONNX_INT8)

        failures: list[str] = []
        for candidate in chain:
            artifact = resolve_versioned_artifact(
                artifacts / "onnx",
                "embedder_int8.onnx"
                if candidate is InferenceBackend.ONNX_INT8
                else "embedder_fp32.onnx",
                version,
            )
            if not artifact.exists():
                failures.append(f"{candidate.value}: artefact not found")
                continue
            try:
                session, input_name = self._create_session(artifact, candidate)
                self._warmup(session, input_name, metadata["image_size"], candidate)
            except Exception as exc:
                failures.append(f"{candidate.value}: {type(exc).__name__}: {exc}")
                continue

            model = LoadedModel(
                name=name,
                version=version,
                task=TaskType.EMBEDDING,
                backend=candidate,
                session=session,
                input_name=input_name,
                class_names=metadata["class_names"],
                wnids=metadata["wnids"],
                image_size=metadata["image_size"],
                artifact_path=artifact,
                loaded_at=datetime.now(UTC),
                metrics={},
                degraded_from=preferred if candidate is not preferred else None,
            )
            self.register(model, make_active=make_active)
            logger.info("model_loaded", model=model.key, backend=candidate.value, task="embedding")
            return model

        raise ModelUnavailableError(
            "No inference backend could be initialised for the embedder.",
            details={"attempts": failures},
        )

    def load_detector(
        self,
        *,
        name: str = "rtdetr-coco-detector",
        version: str = DEFAULT_VERSION,
        make_active: bool = True,
    ) -> LoadedModel:
        """Load the object detector.

        Unlike the classifier, a missing detector is not fatal. The service is
        useful without it, so the caller decides whether to treat failure as an
        error; ``/api/v1/detect`` reports 503 with an explanation when the model
        is absent.
        """
        artifacts = self.settings.artifacts_dir
        labels_path = resolve_versioned_metadata(artifacts, "detection_labels.json", version)
        if labels_path is None:
            raise ModelUnavailableError(
                "Detection artefacts not found. Export them with "
                "`python scripts/prepare_artifacts.py --with-detection`.",
                details={
                    "expected": [
                        str(artifacts / "onnx" / version / "detector_fp32.onnx"),
                        str(artifacts / "onnx" / version / "detection_labels.json"),
                    ]
                },
            )

        metadata = json.loads(labels_path.read_text())
        id2label = {int(k): v for k, v in metadata["id2label"].items()}
        class_names = [id2label.get(i, str(i)) for i in range(metadata["num_classes"])]

        preferred = self.settings.inference_backend
        chain = (
            _FALLBACK_CHAINS.get(preferred, (preferred,))
            if self.settings.enable_graceful_degradation
            else (preferred,)
        )
        # INT8 is served only when requested. The calibration study measured an
        # 87% mAP collapse, so it must never be a silent fallback from FP32.
        if preferred is not InferenceBackend.ONNX_INT8:
            chain = tuple(c for c in chain if c is not InferenceBackend.ONNX_INT8)

        failures: list[str] = []
        for candidate in chain:
            artifact = resolve_versioned_artifact(
                artifacts / "onnx",
                "detector_int8.onnx"
                if candidate is InferenceBackend.ONNX_INT8
                else "detector_fp32.onnx",
                version,
            )
            if not artifact.exists():
                failures.append(f"{candidate.value}: artefact not found")
                continue
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
