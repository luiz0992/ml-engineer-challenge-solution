#!/usr/bin/env python3
"""Re-verify the exported ONNX artefacts against their PyTorch sources.

The exporters each check equivalence at export time and log a max absolute
difference. That number then appears in the model cards, where it is an
assertion nobody can check: the log line is gone, the figure is hand-copied,
and the artefact it described may have been re-exported since.

This re-measures it against the artefact actually on disk and writes
`benchmarks/export_fidelity.json`, so the claim is reproducible rather than
remembered. It is also a standing guard: an export that silently drifts --
a changed opset, a different torch version, a wrapper that stopped applying
sigmoid -- shows up here as a tolerance failure.

Usage::

    uv run python scripts/verify_exports.py
    uv run python scripts/verify_exports.py --check    # non-zero on regression

Requires the training stack (`uv sync --extra train`), because comparing
against PyTorch means loading PyTorch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("verify_exports")

#: Per-model tolerance on the max absolute difference against PyTorch.
#:
#: The classifier is tightest because it is a plain forward pass in fp32. The
#: detector is looser: RT-DETR's decoder accumulates over 300 queries and six
#: decoder layers, so error compounds further than in a single classification
#: head. These are export-fidelity bounds, not accuracy bounds -- a value above
#: them means the graph no longer computes what the model computes.
TOLERANCES = {
    "classifier": 1e-4,
    "embedder": 1e-4,
    "detector": 1e-3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--runs-dir", type=Path, default=Path("models/artifacts/runs"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/export_fidelity.json"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any model exceeds its tolerance",
    )
    return parser


def find_latest_run(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.glob("*") if (p / "config.json").is_file()]
    if not candidates:
        raise SystemExit(
            f"No training runs under {runs_dir}. Train a model first:\n"
            f"  uv run python -m models.training.train"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def verify_classifier(run_dir: Path, onnx_path: Path) -> dict[str, Any]:
    from models.optimisation.export import load_classifier_from_run, verify_onnx_equivalence

    model, metadata = load_classifier_from_run(run_dir)
    max_diff = verify_onnx_equivalence(model, onnx_path, image_size=metadata["image_size"])
    return {"max_abs_diff": max_diff, "compared_against": "timm forward pass"}


def verify_embedder(run_dir: Path, onnx_path: Path) -> dict[str, Any]:
    from models.optimisation.export_embedding import (
        _verify_embeddings,
        build_embedding_model,
        build_normalised_embedder,
    )

    backbone, metadata = build_embedding_model(run_dir)
    # The *wrapper*, not the bare backbone: the exported graph normalises,
    # so comparing against un-normalised features measures the wrapper's
    # absence rather than the export's fidelity.
    wrapper = build_normalised_embedder(backbone).eval()
    max_diff = _verify_embeddings(wrapper, onnx_path, metadata["image_size"])
    return {
        "max_abs_diff": max_diff,
        "compared_against": "backbone with L2 normalisation",
        "embedding_dim": metadata.get("embedding_dim"),
    }


def verify_detector(onnx_path: Path) -> dict[str, Any]:
    from transformers import RTDetrForObjectDetection

    from models.optimisation.export_detection import (
        DEFAULT_DETECTION_MODEL,
        RTDetrExportWrapper,
        _verify,
    )

    model = RTDetrForObjectDetection.from_pretrained(DEFAULT_DETECTION_MODEL)
    model.eval()
    max_diff = _verify(RTDetrExportWrapper(model).eval(), onnx_path)
    return {
        "max_abs_diff": max_diff,
        "compared_against": f"{DEFAULT_DETECTION_MODEL} with sigmoid applied",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    onnx_dir = args.artifacts_dir / "onnx"
    if not onnx_dir.is_dir():
        raise SystemExit(
            f"No artefacts under {onnx_dir}. Build them with:\n"
            f"  uv run python scripts/prepare_artifacts.py --all-models"
        )

    run_dir = find_latest_run(args.runs_dir)
    logger.info("Run: %s", run_dir)

    checks: dict[str, Any] = {}
    for name, verify in (
        ("classifier", lambda: verify_classifier(run_dir, onnx_dir / "classifier_fp32.onnx")),
        ("embedder", lambda: verify_embedder(run_dir, onnx_dir / "embedder_fp32.onnx")),
        ("detector", lambda: verify_detector(onnx_dir / "detector_fp32.onnx")),
    ):
        path = onnx_dir / f"{name}_fp32.onnx"
        if not path.exists():
            logger.warning("%-11s skipped: %s not found", name, path)
            continue

        try:
            result = verify()
        except Exception as exc:
            logger.warning("%-11s FAILED: %s: %s", name, type(exc).__name__, str(exc)[:160])
            checks[name] = {
                "error": f"{type(exc).__name__}: {exc}"[:200],
                "within_tolerance": False,
            }
            continue

        tolerance = TOLERANCES[name]
        result["tolerance"] = tolerance
        result["within_tolerance"] = result["max_abs_diff"] <= tolerance
        result["artifact_mb"] = round(path.stat().st_size / 1e6, 1)
        checks[name] = result

        logger.info(
            "%-11s max |torch - onnx| = %.3e  (tolerance %.0e)  %s",
            name,
            result["max_abs_diff"],
            tolerance,
            "OK" if result["within_tolerance"] else "OUT OF TOLERANCE",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2) + "\n")
    logger.info("Wrote %s", args.output)

    if not checks:
        # An empty report is not a pass. Without this, --check succeeds on a
        # machine with no artefacts at all, which is the one situation where
        # the guard has verified nothing.
        logger.error("No artefacts were verified; nothing under %s", onnx_dir)
        return 1 if args.check else 0

    failed = [name for name, entry in checks.items() if not entry.get("within_tolerance")]
    if failed:
        logger.error("Out of tolerance: %s", ", ".join(failed))
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
