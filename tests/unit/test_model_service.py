"""Tests for model loading, versioning, and backend fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from api.config import InferenceBackend, Settings
from api.exceptions import ModelNotFoundError, ModelUnavailableError
from api.models.schemas import TaskType
from api.services.model_service import ModelService

pytestmark = pytest.mark.unit


class TestLoading:
    def test_loads_and_registers_a_model(self, settings: Settings, artifacts_dir: Path) -> None:
        service = ModelService(settings)
        model = service.load_classifier()

        assert model.name == "tiny-imagenet-classifier"
        assert model.num_classes == 10
        assert model.task is TaskType.CLASSIFICATION
        assert service.is_ready

    def test_reads_labels_from_artifacts_not_the_dataset(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        """Labels travel with the model.

        The serving host may not have the dataset at all, and re-deriving the
        mapping from directory listings would silently permute every label if
        the two ever differed.
        """
        model = ModelService(settings).load_classifier()

        assert model.class_names[0] == "class_0"
        assert model.wnids[0] == "n00000000"
        assert len(model.class_names) == len(model.wnids)

    def test_reports_provenance(self, settings: Settings, artifacts_dir: Path) -> None:
        model = ModelService(settings).load_classifier()

        assert model.backend is InferenceBackend.ONNX
        assert model.artifact_path.exists()
        # Checked in bytes: the synthetic test model is a few KB and rounds to
        # 0.0 MB, which would make a megabyte assertion fail for the wrong reason.
        assert model.artifact_path.stat().st_size > 0
        assert model.artifact_size_mb >= 0
        assert model.loaded_at is not None

    def test_surfaces_training_metrics(self, settings: Settings, artifacts_dir: Path) -> None:
        assert ModelService(settings).load_classifier().metrics["final_acc_top1"] == 0.85

    def test_missing_labels_fails_with_instructions(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        """An unusable artifacts directory must say how to fix it."""
        (artifacts_dir / "labels.json").unlink()

        with pytest.raises(ModelUnavailableError) as exc_info:
            ModelService(settings).load_classifier()

        assert "prepare_artifacts" in exc_info.value.message

    def test_missing_artifacts_lists_what_was_attempted(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        (artifacts_dir / "onnx" / "classifier_fp32.onnx").unlink()

        with pytest.raises(ModelUnavailableError) as exc_info:
            ModelService(settings).load_classifier()

        # Every attempted backend is reported, so the failure is diagnosable
        # without reading the source.
        assert exc_info.value.details["attempts"]


class TestWarmup:
    def test_a_backend_that_cannot_execute_is_rejected(
        self, settings: Settings, artifacts_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session creation succeeding does not mean the backend works.

        ONNX Runtime's CUDA provider resolves cuDNN lazily, so a session builds
        cleanly, passes a provider check, and then fails on every request. The
        warmup inference is what turns that permanent runtime failure into a
        load-time rejection, which is what makes the fallback chain real.
        """

        def _explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("cuDNN is unavailable for the CUDA Execution Provider")

        monkeypatch.setattr(ModelService, "_warmup", staticmethod(_explode))

        with pytest.raises(ModelUnavailableError) as exc_info:
            ModelService(settings).load_classifier()

        assert any("cuDNN" in attempt for attempt in exc_info.value.details["attempts"]), (
            "the underlying failure must be reported, not swallowed"
        )


class TestGracefulDegradation:
    def test_falls_back_when_the_preferred_backend_is_unavailable(
        self,
        settings: Settings,
        artifacts_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TensorRT is requested but cannot initialise.

        The failure is injected rather than relying on TensorRT being absent:
        this development machine has it installed and working, so a test that
        assumed unavailability would pass or fail according to the host rather
        than according to the behaviour under test.

        The service must load on a lower backend rather than refuse to start,
        and must record that it did.
        """
        settings = settings.model_copy(update={"inference_backend": InferenceBackend.TENSORRT})

        original = ModelService._create_session

        def _fail_tensorrt(self: ModelService, artifact: Path, backend: InferenceBackend):
            if backend is InferenceBackend.TENSORRT:
                raise RuntimeError("TensorRT engine build failed")
            return original(self, artifact, backend)

        monkeypatch.setattr(ModelService, "_create_session", _fail_tensorrt)

        model = ModelService(settings).load_classifier()

        assert model.backend is not InferenceBackend.TENSORRT
        # The degradation is recorded, so /health and /models can surface it
        # rather than the service quietly pretending to be configured correctly.
        assert model.degraded_from is InferenceBackend.TENSORRT

    def test_no_fallback_when_degradation_is_disabled(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        """Some deployments must fail rather than serve on a slower backend."""
        settings = settings.model_copy(
            update={
                "inference_backend": InferenceBackend.ONNX_INT8,
                "enable_graceful_degradation": False,
            }
        )

        with pytest.raises(ModelUnavailableError):
            ModelService(settings).load_classifier()


class TestVersioning:
    def test_active_version_serves_unpinned_requests(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        service = ModelService(settings)
        service.load_classifier(version="v1")

        assert service.get_classifier().version == "v1"

    def test_clients_can_pin_a_version(self, settings: Settings, artifacts_dir: Path) -> None:
        service = ModelService(settings)
        service.load_classifier(version="v1")
        service.load_classifier(version="v2", make_active=True)

        assert service.get_classifier().version == "v2"
        assert service.get_classifier("v1").version == "v1"

    def test_promotion_switches_unpinned_traffic(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        service = ModelService(settings)
        service.load_classifier(version="v1")
        service.load_classifier(version="v2", make_active=False)
        assert service.get_classifier().version == "v1"

        service.set_active("tiny-imagenet-classifier", "v2")
        assert service.get_classifier().version == "v2"

    def test_cannot_activate_a_version_that_never_loaded(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        """Promotion must not point traffic at an uninitialised artefact."""
        service = ModelService(settings)
        service.load_classifier(version="v1")

        with pytest.raises(ModelNotFoundError):
            service.set_active("tiny-imagenet-classifier", "v99")

    def test_unknown_version_lists_what_exists(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        service = ModelService(settings)
        service.load_classifier(version="v1")

        with pytest.raises(ModelNotFoundError) as exc_info:
            service.get_classifier("v99")

        assert "v1" in exc_info.value.details["available_versions"]

    def test_unknown_model_is_reported(self, settings: Settings) -> None:
        with pytest.raises(ModelNotFoundError):
            ModelService(settings).get("no-such-model")


class TestInference:
    def test_produces_logits_of_the_expected_shape(self, model_service: ModelService) -> None:
        model = model_service.get_classifier()
        batch = np.zeros((3, 3, 224, 224), dtype=np.float32)

        logits = model_service.run(model, batch)

        assert logits.shape == (3, model.num_classes)
        assert np.isfinite(logits).all()

    def test_dynamic_batch_axis_works(self, model_service: ModelService) -> None:
        """Batch size must not have been folded into a constant at export."""
        model = model_service.get_classifier()

        for batch_size in (1, 2, 7):
            batch = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
            assert model_service.run(model, batch).shape[0] == batch_size

    def test_malformed_input_raises_a_service_error(self, model_service: ModelService) -> None:
        """Runtime failures surface as 503, not an unhandled exception."""
        model = model_service.get_classifier()
        wrong_shape = np.zeros((1, 1, 32, 32), dtype=np.float32)

        with pytest.raises(ModelUnavailableError):
            model_service.run(model, wrong_shape)
