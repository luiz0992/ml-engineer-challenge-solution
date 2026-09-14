"""Export trained models to ONNX and verify numerical equivalence.

Export is only useful if the exported graph computes the same function as the
source model, so :func:`export_onnx` always validates its output against the
PyTorch reference rather than trusting that the conversion succeeded. A silent
numerical divergence here would surface as unexplained accuracy loss in
production, long after the cause.

Design notes
------------
* **Dynamic batch axis.** The serving API handles both single images and
  batches, so batch size is exported as a dynamic dimension. Spatial dimensions
  stay fixed: the preprocessing pipeline always emits 224x224, and a fixed
  shape lets downstream runtimes select better kernels.
* **Opset 18.** torch's exporter refuses to emit below 18 and silently
  upgrades lower requests, so 18 is set explicitly rather than being reached by
  accident. It is supported by ONNX Runtime 1.20+ and TensorRT 10, and provides
  ``LayerNormalization`` as a single fused op rather than the decomposed
  subgraph earlier opsets produce — which matters for transformer throughput.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_OPSET = 18

#: Tolerances for the post-export equivalence check. ONNX Runtime and PyTorch
#: order floating-point reductions differently, so bitwise equality is not
#: achievable; 1e-4 is tight enough to catch a genuinely wrong graph while
#: tolerating benign reassociation.
ATOL = 1e-4
RTOL = 1e-3


@dataclass(slots=True)
class ExportMetadata:
    """Provenance recorded alongside an exported artefact.

    Written next to the model so a deployed artefact can always be traced back
    to the code and configuration that produced it. Without this, an ONNX file
    in a registry is unattributable.
    """

    model_name: str
    task: str
    num_classes: int
    image_size: int
    opset: int
    input_name: str
    output_name: str
    dynamic_batch: bool
    normalisation_mean: tuple[float, float, float]
    normalisation_std: tuple[float, float, float]
    source_checkpoint: str
    torch_version: str
    onnx_version: str
    max_abs_diff: float | None = None

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


class ExportError(RuntimeError):
    """Raised when export fails or the exported graph diverges from the source."""


def _consolidate_external_data(onnx_path: Path) -> None:
    """Inline externally-stored weights into a single ``.onnx`` file.

    torch's exporter writes tensor data to a ``<name>.onnx.data`` sidecar and
    leaves only the graph in the ``.onnx`` file. That split is a deployment
    hazard: copying just the ``.onnx`` produces a model that loads without
    error and then fails at inference, or worse, is silently uninitialised.

    A single self-contained file is safe to move, hash, and cache as one unit.
    ViT-Small is 87 MB, far below the 2 GiB protobuf ceiling, so there is no
    reason to keep the split. Models above that ceiling genuinely require
    external data and would need both files shipped together.
    """
    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if not sidecar.exists():
        return

    # load_external_data=True pulls the sidecar contents into memory.
    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model, str(onnx_path), save_as_external_data=False)
    sidecar.unlink()

    logger.info(
        "Consolidated external weights into a single file (%.1f MB)",
        onnx_path.stat().st_size / 1e6,
    )


def export_onnx(
    model: nn.Module,
    output_path: Path,
    *,
    image_size: int = 224,
    opset: int = DEFAULT_OPSET,
    dynamic_batch: bool = True,
    verify: bool = True,
    device: str = "cpu",
    single_file: bool = True,
) -> float | None:
    """Export ``model`` to ONNX and verify it against the PyTorch reference.

    Returns the maximum absolute difference between PyTorch and ONNX Runtime
    outputs, or ``None`` when ``verify`` is False.

    Exports on CPU by default. A CUDA export can bake device-specific
    behaviour into the graph, and the resulting file should be portable to
    whichever runtime the serving container uses.
    """
    model = model.eval().to(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Batch size 2, not 1: a size-1 batch lets the tracer fold the batch
    # dimension into constants in some operators, producing a graph that
    # silently fails for any other batch size.
    dummy = torch.randn(2, 3, image_size, image_size, device=device)

    dynamic_axes = (
        {"images": {0: "batch_size"}, "logits": {0: "batch_size"}} if dynamic_batch else None
    )

    logger.info("Exporting to ONNX (opset=%d, dynamic_batch=%s)", opset, dynamic_batch)
    try:
        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
        )
    except Exception as exc:
        raise ExportError(f"ONNX export failed: {exc}") from exc

    if single_file:
        _consolidate_external_data(output_path)

    # Structural validation catches malformed graphs that still wrote to disk.
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info(
        "ONNX graph is structurally valid: %s (%.1f MB)",
        output_path,
        output_path.stat().st_size / 1e6,
    )

    if not verify:
        return None

    max_diff = verify_onnx_equivalence(model, output_path, image_size=image_size, device=device)
    logger.info("PyTorch vs ONNX Runtime max |diff| = %.3e", max_diff)

    if max_diff > ATOL:
        raise ExportError(
            f"Exported ONNX graph diverges from the PyTorch model "
            f"(max |diff| = {max_diff:.3e} > {ATOL:.0e}). The export is not "
            f"numerically equivalent and must not be deployed."
        )
    return max_diff


def verify_onnx_equivalence(
    model: nn.Module,
    onnx_path: Path,
    *,
    image_size: int = 224,
    device: str = "cpu",
    batch_sizes: tuple[int, ...] = (1, 3),
    seed: int = 0,
) -> float:
    """Compare PyTorch and ONNX Runtime outputs on random inputs.

    Several batch sizes are checked, including sizes not used during export, to
    confirm the dynamic axis genuinely works rather than having been folded
    into a constant.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    generator = torch.Generator().manual_seed(seed)
    model = model.eval().to(device)
    max_diff = 0.0

    for batch_size in batch_sizes:
        sample = torch.randn(batch_size, 3, image_size, image_size, generator=generator)

        with torch.no_grad():
            torch_out = model(sample.to(device)).cpu().numpy()

        onnx_out = session.run(None, {input_name: sample.numpy()})[0]

        if torch_out.shape != onnx_out.shape:
            raise ExportError(
                f"Shape mismatch at batch_size={batch_size}: "
                f"PyTorch {torch_out.shape} vs ONNX {onnx_out.shape}"
            )

        max_diff = max(max_diff, float(np.abs(torch_out - onnx_out).max()))

    return max_diff


def load_classifier_from_run(
    run_dir: Path,
    *,
    epoch: str | int = "latest",
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a trained classifier from a training run directory.

    Reads the architecture from the run's ``config.json`` rather than taking it
    as an argument, so the model is always reconstructed exactly as trained.
    Weights are loaded from safetensors with ``strict=True``: a silently
    partial load would produce a model with randomly initialised layers that
    still runs and still returns plausible-looking predictions.
    """
    import timm
    from safetensors.torch import load_file

    config = json.loads((run_dir / "config.json").read_text())
    labels = json.loads((run_dir / "labels.json").read_text())

    weights_root = run_dir / "accelerate_states" / "weights"
    if epoch == "latest":
        candidates = sorted(weights_root.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
        if not candidates:
            raise ExportError(f"No weight directories under {weights_root}")
        weights_dir = candidates[-1]
    else:
        weights_dir = weights_root / f"epoch_{epoch}"
        if not weights_dir.is_dir():
            raise ExportError(f"No weights for epoch {epoch} at {weights_dir}")

    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = load_file(weights_dir / "model.safetensors")
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    logger.info("Loaded %s from %s (%d classes)", model_name, weights_dir, num_classes)

    metadata = {
        "model_name": model_name,
        "num_classes": num_classes,
        "image_size": config["dataloader"]["image_size"],
        "class_names": labels["class_names"],
        "wnids": labels["wnids"],
        "source_checkpoint": str(weights_dir),
        "normalisation": config["dataloader"]["normalisation"],
    }
    return model, metadata
