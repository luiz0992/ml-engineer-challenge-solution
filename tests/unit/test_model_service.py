"""Tests for model loading, versioning, and backend fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from api.config import InferenceBackend, Settings
from api.exceptions import ModelNotFoundError, ModelUnavailableError
from api.models.schemas import TaskType
from api.services import runtime
from api.services.model_service import (
    ModelService,
)
from api.services.model_service import discover_versions as runtime_discover
from api.services.model_service import resolve_versioned_artifact as runtime_versioning

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

    def test_int8_is_served_when_requested_and_present(
        self, settings: Settings, artifacts_dir: Path, synthetic_onnx: Path
    ) -> None:
        """INT8 is a live backend, not a dead artefact.

        The default path still serves FP32 because the calibration study
        measured a 5.4pp drop. Requesting INT8 explicitly must load that
        graph rather than ignoring the setting.
        """
        import shutil

        shutil.copy2(synthetic_onnx, artifacts_dir / "onnx" / "classifier_int8.onnx")
        int8_settings = settings.model_copy(
            update={"inference_backend": InferenceBackend.ONNX_INT8}
        )

        model = ModelService(int8_settings).load_classifier()

        assert model.backend is InferenceBackend.ONNX_INT8
        assert model.artifact_path.name == "classifier_int8.onnx"
        assert model.degraded_from is None


class TestVersioning:
    @pytest.fixture
    def two_versions(self, artifacts_dir: Path, synthetic_onnx: Path) -> Path:
        """Materialise a real, self-describing `v2` alongside the default.

        These tests previously just passed `version="v2"` to a loader that
        ignored it, so they asserted that pinning worked while the service was
        quietly loading v1's file. A second version now has to exist on disk,
        carrying its own labels — which is what a deployment would ship, and
        what stops v2's predictions being named from v1's class list.
        """
        import shutil

        versioned = artifacts_dir / "onnx" / "v2"
        versioned.mkdir(parents=True, exist_ok=True)
        shutil.copy(synthetic_onnx, versioned / "classifier_fp32.onnx")
        shutil.copy(artifacts_dir / "labels.json", versioned / "labels.json")
        return artifacts_dir

    def test_active_version_serves_unpinned_requests(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        service = ModelService(settings)
        service.load_classifier(version="v1")

        assert service.get_classifier().version == "v1"

    def test_clients_can_pin_a_version(self, settings: Settings, two_versions: Path) -> None:
        service = ModelService(settings)
        service.load_classifier(version="v1")
        service.load_classifier(version="v2", make_active=True)

        assert service.get_classifier().version == "v2"
        assert service.get_classifier("v1").version == "v1"

    def test_promotion_switches_unpinned_traffic(
        self, settings: Settings, two_versions: Path
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


class TestCpuBudget:
    """`available_cpus` must respect the cgroup quota, not the host core count.

    ONNX Runtime sizes its intra-op thread pool from the number it is given. In
    a CPU-limited container `os.cpu_count()` reports the host's processors, so
    the default pool oversubscribes the quota and the container spends its time
    context-switching. Measured on this project's image: 102.8 ms per
    classification with the default 32 threads against 12.4 ms with 4, matching
    a 4-CPU quota.

    The failure is silent -- the container is healthy and correct, just 8x
    slower -- which is why it is worth a test rather than a comment.
    """

    @pytest.fixture
    def many_host_cpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin the affinity count so the quota is what the test measures.

        `available_cpus` takes the minimum of affinity and quota. Without this
        the result depends on the machine: these assertions passed on a 32-core
        workstation and failed on a 2-core CI runner, where the affinity floor
        was lower than the quota under test.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(64)))

    def test_reads_a_cgroup_v2_quota(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, many_host_cpus: None
    ) -> None:
        quota = tmp_path / "cpu.max"
        quota.write_text("400000 100000")
        monkeypatch.setattr("api.services.runtime._CGROUP_V2_QUOTA", quota)
        monkeypatch.setattr("api.services.runtime._CGROUP_V1_QUOTA", tmp_path / "absent")

        assert runtime.available_cpus() == 4

    def test_unlimited_v2_quota_falls_back_to_affinity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        quota = tmp_path / "cpu.max"
        quota.write_text("max 100000")
        monkeypatch.setattr("api.services.runtime._CGROUP_V2_QUOTA", quota)
        monkeypatch.setattr("api.services.runtime._CGROUP_V1_QUOTA", tmp_path / "absent")

        assert runtime.available_cpus() == len(os.sched_getaffinity(0))

    def test_reads_a_cgroup_v1_quota(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, many_host_cpus: None
    ) -> None:
        (tmp_path / "cpu.cfs_quota_us").write_text("200000")
        (tmp_path / "cpu.cfs_period_us").write_text("100000")
        monkeypatch.setattr("api.services.runtime._CGROUP_V2_QUOTA", tmp_path / "absent")
        monkeypatch.setattr("api.services.runtime._CGROUP_V1_QUOTA", tmp_path / "cpu.cfs_quota_us")
        monkeypatch.setattr(
            "api.services.runtime._CGROUP_V1_PERIOD", tmp_path / "cpu.cfs_period_us"
        )

        assert runtime.available_cpus() == 2

    def test_the_quota_never_exceeds_cpu_affinity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A generous quota must not licence more threads than we can run.

        A pinned process is bounded by affinity even with no quota set, so the
        budget is the minimum of the two.
        """
        quota = tmp_path / "cpu.max"
        quota.write_text("6400000 100000")
        monkeypatch.setattr("api.services.runtime._CGROUP_V2_QUOTA", quota)
        monkeypatch.setattr("api.services.runtime._CGROUP_V1_QUOTA", tmp_path / "absent")
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2})

        assert runtime.available_cpus() == 3

    def test_a_corrupt_quota_file_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cgroup we cannot parse must degrade, not take the service down."""
        quota = tmp_path / "cpu.max"
        quota.write_text("garbage")
        monkeypatch.setattr("api.services.runtime._CGROUP_V2_QUOTA", quota)
        monkeypatch.setattr("api.services.runtime._CGROUP_V1_QUOTA", tmp_path / "absent")

        assert runtime.available_cpus() >= 1

    def test_never_returns_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sub-1.0 CPU quota must floor at one thread, not zero.

        `intra_op_num_threads=0` tells ONNX Runtime to pick the default, which
        is the host count -- the exact bug this function exists to avoid, and
        it would reappear for any container given less than a full CPU.
        """
        quota = tmp_path / "cpu.max"
        quota.write_text("50000 100000")
        monkeypatch.setattr("api.services.runtime._CGROUP_V2_QUOTA", quota)
        monkeypatch.setattr("api.services.runtime._CGROUP_V1_QUOTA", tmp_path / "absent")

        assert runtime.available_cpus() == 1

    def test_sessions_are_built_with_a_bounded_thread_pool(self, synthetic_onnx: Path) -> None:
        """The budget must actually reach the session, not just be computed."""
        session, _ = runtime.create_session(str(synthetic_onnx))

        assert session.get_session_options().intra_op_num_threads == runtime.available_cpus()
        assert session.get_session_options().inter_op_num_threads == 1


class TestVersionedArtifacts:
    """Resolving a pinned version to the file that actually holds its weights.

    Version pinning is only a contract if a pinned version either serves that
    version's weights or fails. Serving the default's weights under another
    version's name is the worst of the three outcomes: the response, the
    provenance block, the audit row and the A/B analysis would all agree on a
    version that never ran.
    """

    def test_versioned_directory_wins(self, tmp_path: Path) -> None:
        onnx = tmp_path / "onnx"
        (onnx / "v2").mkdir(parents=True)
        (onnx / "classifier_fp32.onnx").write_bytes(b"default")
        (onnx / "v2" / "classifier_fp32.onnx").write_bytes(b"v2")

        resolved = runtime_versioning(onnx, "classifier_fp32.onnx", "v2")

        assert resolved == onnx / "v2" / "classifier_fp32.onnx"
        assert resolved.read_bytes() == b"v2"

    def test_default_falls_back_to_the_flat_layout(self, tmp_path: Path) -> None:
        """The layout the artefact pipeline produces today must keep working."""
        onnx = tmp_path / "onnx"
        onnx.mkdir(parents=True)
        (onnx / "classifier_fp32.onnx").write_bytes(b"default")

        assert runtime_versioning(onnx, "classifier_fp32.onnx", "v1") == (
            onnx / "classifier_fp32.onnx"
        )

    def test_a_missing_version_never_falls_back_to_the_default(self, tmp_path: Path) -> None:
        """The property this whole function exists for."""
        onnx = tmp_path / "onnx"
        onnx.mkdir(parents=True)
        (onnx / "classifier_fp32.onnx").write_bytes(b"default")

        resolved = runtime_versioning(onnx, "classifier_fp32.onnx", "v7")

        assert resolved == onnx / "v7" / "classifier_fp32.onnx"
        assert not resolved.exists(), "a missing version must fail, not serve v1"

    def test_loading_an_absent_version_raises(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        service = ModelService(settings)

        with pytest.raises(ModelUnavailableError):
            service.load_classifier(version="v9")

    def test_discovers_versions_present_on_disk(self, tmp_path: Path) -> None:
        onnx = tmp_path / "onnx"
        (onnx / "v2").mkdir(parents=True)
        (onnx / "v10").mkdir(parents=True)
        (onnx / "classifier_fp32.onnx").write_bytes(b"default")

        # Natural order, not lexicographic: v10 is a later version than v2.
        assert runtime_discover(onnx) == ["v1", "v2", "v10"]

    def test_ignores_directories_that_are_not_versions(self, tmp_path: Path) -> None:
        """`trt_cache/` lives here, and is not a model version."""
        onnx = tmp_path / "onnx"
        (onnx / "trt_cache").mkdir(parents=True)
        (onnx / "v2.tmp-partial").mkdir(parents=True)
        (onnx / "classifier_fp32.onnx").write_bytes(b"default")

        assert runtime_discover(onnx) == ["v1"]

    def test_no_artefacts_discovers_nothing(self, tmp_path: Path) -> None:
        onnx = tmp_path / "onnx"
        onnx.mkdir(parents=True)

        assert runtime_discover(onnx) == []

    def test_a_second_version_is_servable_end_to_end(
        self, settings: Settings, artifacts_dir: Path, synthetic_onnx: Path
    ) -> None:
        """Shipping a directory is enough; no code change is required."""
        import shutil

        versioned = artifacts_dir / "onnx" / "v2"
        versioned.mkdir(parents=True, exist_ok=True)
        shutil.copy(synthetic_onnx, versioned / "classifier_fp32.onnx")
        shutil.copy(artifacts_dir / "labels.json", versioned / "labels.json")

        service = ModelService(settings)
        service.load_classifier()
        second = service.load_classifier(version="v2", make_active=False)

        assert second.version == "v2"
        assert second.artifact_path == versioned / "classifier_fp32.onnx"
        # The default must still be what an unpinned caller gets.
        assert service.get_classifier().version == "v1"
        assert service.get_classifier("v2").version == "v2"


class TestProviderHonesty:
    """A reported backend must be the backend that is running.

    The provider check used to compare against the *filtered* provider list.
    If the requested provider was not registered in this ONNX Runtime build it
    was filtered out, CPU was appended, and the comparison then succeeded
    trivially — so the guard passed in exactly the case it existed to catch.
    The API would advertise `backend=tensorrt` on every response, health check
    and audit row while running on CPU two orders of magnitude slower.
    """

    def test_requesting_an_unregistered_provider_raises(
        self, synthetic_onnx: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

        with pytest.raises(RuntimeError, match="does not satisfy"):
            runtime.create_session(
                str(synthetic_onnx),
                providers=["TensorrtExecutionProvider", "CPUExecutionProvider"],
            )

    def test_a_cpu_request_is_unaffected(
        self, synthetic_onnx: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        session, _ = runtime.create_session(str(synthetic_onnx))

        assert session.get_providers() == ["CPUExecutionProvider"]

    def test_the_fallback_chain_records_the_degradation(
        self,
        settings: Settings,
        artifacts_dir: Path,
        synthetic_onnx: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unavailable preferred backend must degrade *and say so*.

        This is the behaviour the `ModelRunningDegraded` alert watches for, and
        it only works if the session constructor refuses to pretend.

        It should land on `onnx`, one step down, not tumble all the way to
        INT8: CPU satisfies `onnx`, so the chain stops there rather than paying
        5.4 points of top-1 accuracy to escape a degradation that has already
        been escaped.
        """
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        tensorrt_settings = settings.model_copy(
            update={"inference_backend": InferenceBackend.TENSORRT}
        )

        model = ModelService(tensorrt_settings).load_classifier()

        assert model.backend is InferenceBackend.ONNX
        assert model.degraded_from is InferenceBackend.TENSORRT

    def test_a_cpu_only_deployment_is_not_degraded(
        self, settings: Settings, artifacts_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default configuration on a CPU host must report itself healthy.

        `InferenceBackend.ONNX` offers ONNX Runtime CUDA *then* CPU, so a
        CPU-only host satisfies it -- it is running what it was built for, not
        a fallback. Treating CPU as a degradation here marked every CPU
        deployment degraded, held `ModelRunningDegraded` permanently red, and
        pushed serving down to INT8. An alert that can never be green is as
        useless as one that can never fire.
        """
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

        model = ModelService(settings).load_classifier()

        assert model.backend is InferenceBackend.ONNX
        assert model.degraded_from is None


class TestVersionedMetadata:
    """A version's labels must come from that version.

    Versioning the weights without versioning the label map means a `v2` built
    on a re-ordered or extended class list is served with `v1`'s names. Every
    prediction is mislabelled, silently, and the output still looks like a
    plausible class — the same failure as serving the wrong weights, and harder
    to spot.
    """

    def _make_v2(self, artifacts_dir: Path, synthetic_onnx: Path, *, labels: bool) -> Path:
        import shutil

        versioned = artifacts_dir / "onnx" / "v2"
        versioned.mkdir(parents=True, exist_ok=True)
        shutil.copy(synthetic_onnx, versioned / "classifier_fp32.onnx")

        if labels:
            original = json.loads((artifacts_dir / "labels.json").read_text())
            original["class_names"] = [f"v2-{name}" for name in original["class_names"]]
            (versioned / "labels.json").write_text(json.dumps(original))
        return versioned

    def test_a_version_without_its_own_labels_is_refused(
        self, settings: Settings, artifacts_dir: Path, synthetic_onnx: Path
    ) -> None:
        self._make_v2(artifacts_dir, synthetic_onnx, labels=False)
        service = ModelService(settings)

        with pytest.raises(ModelUnavailableError, match="labels"):
            service.load_classifier(version="v2", make_active=False)

    def test_each_version_uses_its_own_labels(
        self, settings: Settings, artifacts_dir: Path, synthetic_onnx: Path
    ) -> None:
        self._make_v2(artifacts_dir, synthetic_onnx, labels=True)
        service = ModelService(settings)

        v1 = service.load_classifier()
        v2 = service.load_classifier(version="v2", make_active=False)

        assert not v1.class_names[0].startswith("v2-")
        assert v2.class_names[0].startswith("v2-"), "v2 was named from another version's labels"

    def test_the_default_version_still_reads_the_flat_layout(
        self, settings: Settings, artifacts_dir: Path
    ) -> None:
        """The layout the artefact pipeline produces today must keep working."""
        model = ModelService(settings).load_classifier()

        assert model.class_names
