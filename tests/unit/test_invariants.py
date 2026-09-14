"""Guards against defect classes that have already recurred in this project.

These tests do not exercise behaviour. They assert structural properties of the
codebase, because two specific mistakes proved able to reappear each time a new
call site was added, and a comment asking future contributors to remember is not
a control.

Both failures share a shape: **code that succeeds while doing nothing.** That is
the hardest class of defect to notice, because everything looks healthy.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Modules allowed to construct a session directly. `runtime` defines the
#: shared helper; `benchmark` needs per-provider options the helper does not
#: accept, and applies the preload and provider assertion itself.
SESSION_CONSTRUCTION_ALLOWED = {
    "api/services/runtime.py",
    "models/optimisation/benchmark.py",
}

#: Providers whose libraries resolve lazily, so a session can be created
#: successfully and then fail on every inference.
GPU_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")


def _source_files() -> list[Path]:
    roots = ("api", "models", "scripts", "worker")
    return [
        path
        for root in roots
        for path in (REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    ]


class TestGpuSessionConstruction:
    """ONNX Runtime sessions requesting a GPU must go through the helper.

    ONNX Runtime resolves cuDNN lazily at the first kernel launch. A session
    requesting ``CUDAExecutionProvider`` is therefore created successfully,
    reports the provider it was asked for, and then fails *every* inference
    with ``NOT_IMPLEMENTED``. TensorRT fails differently but equally quietly:
    it falls back to CPU without raising, so the service runs about a hundred
    times slower than its logs claim.

    This recurred three times -- in the model service, the benchmark harness,
    and the evaluation scripts -- because each new call site had to remember to
    preload. :func:`api.services.runtime.create_session` removes the
    opportunity to forget, and this test removes the opportunity to bypass it.
    """

    def test_gpu_sessions_use_the_shared_constructor(self) -> None:
        violations: list[str] = []

        for path in _source_files():
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative in SESSION_CONSTRUCTION_ALLOWED:
                continue

            source = path.read_text()
            if "InferenceSession" not in source:
                continue

            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name != "InferenceSession":
                    continue

                # A CPU-only session is safe: there are no lazily-resolved
                # libraries to fail on.
                call_source = ast.get_source_segment(source, node) or ""
                if any(provider in call_source for provider in GPU_PROVIDERS):
                    violations.append(f"{relative}:{node.lineno}")

        assert not violations, (
            "These call sites construct a GPU InferenceSession directly instead "
            "of using api.services.runtime.create_session:\n  "
            + "\n  ".join(violations)
            + "\n\nA directly-constructed CUDA session is created successfully and "
            "then fails every inference, because cuDNN resolves lazily. Use "
            "create_session, which preloads the libraries and verifies the "
            "provider actually took effect."
        )

    def test_the_shared_constructor_still_preloads(self) -> None:
        """A regression guard on the helper itself.

        If the preload were removed from ``create_session``, every call site
        would silently inherit the original bug and the test above would still
        pass.
        """
        source = (REPO_ROOT / "api/services/runtime.py").read_text()

        assert "preload_cuda_libraries()" in source
        assert "preload_tensorrt_libraries()" in source
        assert "_PRELOADED" in source


class TestAlertsAreFireable:
    """Every alert must reference a metric something actually exports.

    An alert on a metric that is never published sits permanently green. That
    is worse than having no alert, because it looks like coverage and stops
    anyone asking whether the condition is monitored.

    This was not hypothetical: ``AuditDefaultPartitionNonEmpty`` was written
    against ``inference_logs_default_partition_rows``, which nothing exported.
    """

    @staticmethod
    def _exported_metrics() -> set[str]:
        """Collect metric names defined anywhere in the codebase."""
        names: set[str] = set()

        for path in _source_files():
            source = path.read_text()
            # prometheus_client constructors take the metric name first.
            for match in re.finditer(
                r'\b(?:Counter|Gauge|Histogram|Summary)\(\s*["\']([a-zA-Z_:][a-zA-Z0-9_:]*)["\']',
                source,
            ):
                names.add(match.group(1))

        # Emitted by Compose-defined jobs rather than application code. Listed
        # explicitly so adding one is a deliberate act.
        names.update(
            {
                "inference_logs_default_partition_rows",
                "model_drift_severity",
                "model_drift_statistic",
            }
        )
        # Provided by Prometheus and the pushgateway themselves.
        names.update({"up", "push_time_seconds"})
        return names

    @staticmethod
    def _alert_metrics() -> dict[str, set[str]]:
        """Extract the metric names each alert expression references."""
        rules = yaml.safe_load((REPO_ROOT / "monitoring/prometheus/alerts.yml").read_text())

        # PromQL keywords and functions that look like identifiers.
        reserved = {
            "sum",
            "rate",
            "avg",
            "min",
            "max",
            "count",
            "by",
            "without",
            "on",
            "group_left",
            "group_right",
            "increase",
            "histogram_quantile",
            "clamp_min",
            "clamp_max",
            "time",
            "le",
            "and",
            "or",
            "unless",
            "offset",
            "bool",
            "irate",
            "delta",
            "abs",
            "ceil",
            "floor",
            "round",
            "topk",
            "bottomk",
            "quantile",
            "stddev",
            "job",
            "instance",
        }

        found: dict[str, set[str]] = {}
        for group in rules["groups"]:
            for rule in group["rules"]:
                expression = rule["expr"]
                # Strip label selectors so label values are not mistaken for
                # metric names.
                stripped = re.sub(r"\{[^}]*\}", "", expression)
                metrics = {
                    token
                    for token in re.findall(r"\b[a-zA-Z_:][a-zA-Z0-9_:]*\b", stripped)
                    if token not in reserved and not token.isdigit()
                }
                found[rule["alert"]] = metrics
        return found

    def test_every_alert_references_an_exported_metric(self) -> None:
        exported = self._exported_metrics()
        unfireable: list[str] = []

        for alert, metrics in self._alert_metrics().items():
            # Histogram queries reference the generated _bucket/_sum/_count
            # series, which prometheus_client derives from the base name.
            resolvable = {
                metric
                for metric in metrics
                if metric in exported
                or re.sub(r"_(bucket|sum|count|total)$", "", metric) in exported
                or metric.replace("_total", "") in exported
            }
            if not resolvable:
                unfireable.append(f"{alert} -> {sorted(metrics)}")

        assert not unfireable, (
            "These alerts reference no metric that anything exports, so they can "
            "never fire:\n  "
            + "\n  ".join(unfireable)
            + "\n\nAn alert that cannot fire is worse than no alert: it looks like "
            "coverage and stops anyone asking whether the condition is monitored."
        )

    def test_every_alert_explains_what_to_do(self) -> None:
        """An alert without an action is noise, and noisy alerts get muted."""
        rules = yaml.safe_load((REPO_ROOT / "monitoring/prometheus/alerts.yml").read_text())

        missing: list[str] = []
        for group in rules["groups"]:
            for rule in group["rules"]:
                description = rule.get("annotations", {}).get("description", "")
                if len(description.strip()) < 40:
                    missing.append(rule["alert"])

        assert not missing, f"Alerts with no actionable description: {missing}"

    def test_alerts_are_valid_promql_shape(self) -> None:
        """Catch unbalanced braces and empty expressions before deployment."""
        rules = yaml.safe_load((REPO_ROOT / "monitoring/prometheus/alerts.yml").read_text())

        for group in rules["groups"]:
            for rule in group["rules"]:
                expression = rule["expr"]
                assert expression.strip(), f"{rule['alert']} has an empty expression"
                assert expression.count("{") == expression.count("}"), (
                    f"{rule['alert']} has unbalanced braces"
                )
                assert expression.count("(") == expression.count(")"), (
                    f"{rule['alert']} has unbalanced parentheses"
                )
