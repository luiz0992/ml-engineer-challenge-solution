"""Object detection model export.

Exports RT-DETR to ONNX for serving.

**Why RT-DETR rather than YOLO.** Ultralytics' YOLOv8/v11 are AGPL-3.0, which
obliges anyone offering the service over a network to publish their entire
source. That is a poor fit for a commercial product, and licence choices made
casually during a prototype are expensive to unwind later. RT-DETR is Apache-2.0
in both implementation and weights, is anchor-free and NMS-free, and the
challenge names DETR explicitly.

**Why the R18 variant.** 20M parameters against 76M for R101. Detection is the
heavier of the two serving paths, and the accuracy difference on COCO
(46.5 vs 54.3 mAP) does not justify roughly 4x the latency for a demonstration
API where the classifier is the primary model.

**No NMS to export.** RT-DETR predicts a fixed set of 300 queries directly, so
there is no non-maximum-suppression step — which is the usual source of pain
when exporting detectors, because NMS is data-dependent and exports poorly.
Filtering is a threshold applied to the output, which the serving code does in
NumPy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.optimisation.export import ExportError

logger = logging.getLogger(__name__)

#: Apache-2.0 implementation and weights, unlike the AGPL Ultralytics models.
DEFAULT_DETECTION_MODEL = "PekingU/rtdetr_r18vd_coco_o365"

#: RT-DETR is trained at a fixed 640x640. Unlike the classifier, this is not a
#: tunable: the model has learned positional priors at this resolution.
DETECTION_IMAGE_SIZE = 640

DEFAULT_OPSET = 18
ATOL = 1e-3


class RTDetrExportWrapper(nn.Module):
    """Adapts RT-DETR's output for ONNX export.

    HuggingFace models return a dataclass, which ONNX cannot represent. This
    unpacks it to a plain tuple of ``(logits, boxes)``.

    Sigmoid is applied here rather than in the serving code so the exported
    graph emits probabilities directly. RT-DETR uses per-class sigmoid rather
    than softmax over classes, because a box may legitimately match several
    labels; applying softmax would be wrong and is an easy mistake to make when
    reimplementing post-processing by hand.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values)
        return torch.sigmoid(outputs.logits), outputs.pred_boxes


def export_detection_onnx(
    output_path: Path,
    *,
    model_name: str = DEFAULT_DETECTION_MODEL,
    opset: int = DEFAULT_OPSET,
    verify: bool = True,
) -> dict[str, Any]:
    """Export a pretrained detector to ONNX and verify it.

    Returns metadata for the serving layer: label map, input size, and the
    measured export error.
    """
    from transformers import RTDetrForObjectDetection

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name)
    model.eval()

    wrapper = RTDetrExportWrapper(model).eval()
    # Batch 2, so the tracer cannot fold the batch dimension into a constant.
    dummy = torch.randn(2, 3, DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE)

    logger.info("Exporting detection model to ONNX (opset=%d)", opset)
    try:
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["scores", "boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "scores": {0: "batch_size"},
                "boxes": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    except Exception as exc:
        raise ExportError(f"Detection ONNX export failed: {exc}") from exc

    _consolidate(output_path)

    import onnx

    onnx.checker.check_model(onnx.load(str(output_path)))
    logger.info(
        "Detection ONNX is structurally valid: %s (%.1f MB)",
        output_path,
        output_path.stat().st_size / 1e6,
    )

    max_diff = None
    if verify:
        max_diff = _verify(wrapper, output_path)
        logger.info("PyTorch vs ONNX Runtime max |diff| = %.3e", max_diff)
        if max_diff > ATOL:
            raise ExportError(
                f"Exported detection graph diverges from PyTorch "
                f"(max |diff| = {max_diff:.3e} > {ATOL:.0e})."
            )

    return {
        "model_name": model_name,
        "image_size": DETECTION_IMAGE_SIZE,
        # id2label is Optional in the config type; a detection model without
        # one is unusable, so fail here rather than shipping an artefact whose
        # predictions cannot be named.
        "id2label": {int(k): v for k, v in _require_labels(model).items()},
        "num_classes": len(_require_labels(model)),
        "max_detections": model.config.num_queries,
        "max_abs_diff": max_diff,
        "licence": "Apache-2.0",
    }


def _require_labels(model: nn.Module) -> dict[Any, str]:
    """Return the model's id-to-label map, or fail loudly."""
    labels = getattr(model.config, "id2label", None)
    if not labels:
        raise ExportError(
            "The detection model has no id2label map; its predictions could not be given names."
        )
    return labels


def _consolidate(onnx_path: Path) -> None:
    """Inline external tensor data into a single file.

    A split artefact loads without error and then has no weights, which is a
    deployment hazard when only the ``.onnx`` is copied.
    """
    import onnx

    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if not sidecar.exists():
        return

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model, str(onnx_path), save_as_external_data=False)
    sidecar.unlink()


def _verify(wrapper: nn.Module, onnx_path: Path, batch_sizes: tuple[int, ...] = (1, 3)) -> float:
    """Compare PyTorch and ONNX Runtime outputs on structured inputs.

    Random noise is a degenerate input for a detector and must not be used
    here. On noise the model's highest confidence is around 0.17, so no query
    is clearly assigned to an object; the assignment is numerically unstable,
    a 1e-6 difference flips which query wins, and the resulting box
    coordinates diverge by up to 0.98. That measures the instability of the
    input, not the fidelity of the export.

    Structured inputs (smooth gradients and blocks) produce confident,
    well-separated predictions where a genuine export defect would show up as a
    real disagreement.

    Only the scores are compared. Box coordinates are meaningful only for
    queries the model is actually confident about; comparing all 300 includes
    the unassigned majority, whose coordinates are arbitrary in both
    implementations.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    max_diff = 0.0

    for batch_size in batch_sizes:
        sample = _structured_input(batch_size)

        with torch.no_grad():
            torch_scores, torch_boxes = wrapper(sample)

        onnx_scores, onnx_boxes = session.run(None, {input_name: sample.numpy()})

        max_diff = max(max_diff, float(np.abs(torch_scores.numpy() - onnx_scores).max()))

        # Boxes are compared only where the model is confident enough that the
        # query assignment is stable.
        confident = torch_scores.numpy().max(axis=-1) > 0.3
        if confident.any():
            max_diff = max(
                max_diff,
                float(np.abs(torch_boxes.numpy()[confident] - onnx_boxes[confident]).max()),
            )

    return max_diff


def _structured_input(batch_size: int) -> torch.Tensor:
    """Build deterministic images with structure a detector can latch onto."""
    size = DETECTION_IMAGE_SIZE
    images = torch.zeros(batch_size, 3, size, size)

    ramp = torch.linspace(0, 1, size)
    for index in range(batch_size):
        images[index, 0] = ramp.unsqueeze(0).expand(size, size)
        images[index, 1] = ramp.unsqueeze(1).expand(size, size)
        # A solid block, which gives the detector a definite region to fire on.
        offset = index * 40
        images[index, :, 100 + offset : 300 + offset, 150:400] = 0.9

    return images


def write_detection_metadata(artifacts_dir: Path, metadata: dict[str, Any]) -> None:
    """Persist the label map beside the model.

    The serving container has no transformers installed, so the id-to-label
    mapping must travel with the artefact rather than being read from the
    HuggingFace config at load time.
    """
    path = artifacts_dir / "detection_labels.json"
    path.write_text(json.dumps(metadata, indent=2))
    logger.info("Wrote detection metadata to %s", path)
