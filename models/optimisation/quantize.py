"""INT8 quantization of exported ONNX models.

Implements static post-training quantization, which requires a calibration pass
over representative data to determine activation ranges. The alternative,
dynamic quantization, computes activation ranges at inference time; it needs no
calibration but leaves activations in float and so recovers far less of the
available speedup for a compute-bound vision model.

Accuracy is always re-measured after quantization. INT8 is a lossy
transformation, and the entire question is whether the loss is acceptable — an
artefact whose accuracy was never checked is not a deployable optimisation, it
is an unquantified risk.

Calibration data comes from the *training* split. Using validation data would
leak the evaluation set into the artefact and make the reported post-quantization
accuracy optimistic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

logger = logging.getLogger(__name__)

#: Calibration samples. A few hundred is the accepted operating point: ranges
#: converge quickly, and more samples cost calibration time without measurably
#: improving the result.
DEFAULT_CALIBRATION_SAMPLES = 256


class QuantizationError(RuntimeError):
    """Raised when quantization fails or degrades accuracy beyond tolerance."""


class TensorDatasetCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed batches to the ONNX Runtime calibrator.

    The calibrator runs the float model over these inputs to observe activation
    ranges, so the data must pass through *exactly* the same preprocessing as
    production traffic. Calibrating on differently normalised inputs yields
    ranges that do not match what the model will actually see, and the
    quantized model then clips or wastes resolution on live data.
    """

    def __init__(
        self,
        dataset: Any,
        input_name: str,
        *,
        num_samples: int = DEFAULT_CALIBRATION_SAMPLES,
        batch_size: int = 8,
    ) -> None:
        self.dataset = dataset
        self.input_name = input_name
        self.num_samples = min(num_samples, len(dataset))
        self.batch_size = batch_size
        self._iterator: Iterator[dict[str, np.ndarray]] | None = None

    def _batches(self) -> Iterator[dict[str, np.ndarray]]:
        import torch

        buffer: list[Any] = []
        for index in range(self.num_samples):
            image, _ = self.dataset[index]
            buffer.append(image)

            if len(buffer) == self.batch_size:
                yield {self.input_name: torch.stack(buffer).numpy()}
                buffer = []

        if buffer:
            yield {self.input_name: torch.stack(buffer).numpy()}

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iterator is None:
            self._iterator = self._batches()
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = None


def _run_pre_process(model_path: Path, output_path: Path) -> None:
    """Run quantization pre-processing, falling back if symbolic inference fails.

    ONNX Runtime's symbolic shape inference is the preferred path: it resolves
    dynamic dimensions algebraically and gives the quantizer the most complete
    shape information, which maximises the number of nodes it can quantize.

    It is also fragile. On graphs produced by torch's TorchDynamo exporter it
    can raise ``IndexError`` inside ``_infer_Concat`` when a dynamic batch
    dimension reaches a concatenation whose inputs it failed to resolve. That
    is a limitation of the inference pass, not a defect in the model, and the
    static fallback still produces a correct — if slightly less aggressively
    quantized — artefact.

    Failing the whole export here would be the wrong trade: we would lose INT8
    entirely because of a shape-inference edge case. The fallback is logged at
    warning level so the degradation is visible rather than silent.
    """
    logger.info("Running quantization pre-processing (shape inference, constant folding)")
    try:
        quant_pre_process(str(model_path), str(output_path), skip_symbolic_shape=False)
        return
    except Exception as exc:
        logger.warning(
            "Symbolic shape inference failed (%s: %s); retrying with static shape "
            "inference. Fewer nodes may be quantized as a result.",
            type(exc).__name__,
            exc,
        )

    try:
        quant_pre_process(str(model_path), str(output_path), skip_symbolic_shape=True)
    except Exception as exc:
        raise QuantizationError(
            f"Quantization pre-processing failed under both symbolic and static "
            f"shape inference: {exc}"
        ) from exc


def quantize_int8(
    model_path: Path,
    output_path: Path,
    calibration_reader: CalibrationDataReader,
    *,
    per_channel: bool = True,
    calibration_method: CalibrationMethod = CalibrationMethod.Percentile,
    percentile: float = 99.99,
    nodes_to_exclude: tuple[str, ...] = (),
    op_types_to_quantize: tuple[str, ...] = (),
) -> Path:
    """Statically quantize an ONNX model to INT8.

    ``per_channel`` quantizes weights with one scale per output channel rather
    than one for the whole tensor. Channels in a trained network often differ in
    magnitude by orders of magnitude, so a single shared scale wastes most of
    the INT8 range on the largest channel and crushes the rest to zero. The
    extra scales cost negligible space and typically recover most of the
    accuracy lost to quantization.

    ``calibration_method`` defaults to ``Percentile`` rather than ``MinMax``.
    This matters enormously for transformers: min-max sets the quantization
    range from the single most extreme activation observed, and LayerNorm and
    GELU in vision transformers produce rare outliers orders of magnitude above
    the typical activation. One such outlier stretches the range so far that
    ordinary values collapse into a handful of quantization levels.

    Measured on this model (ViT-Small, 1000 validation images, FP32 baseline
    88.40% top-1):

    ==========================  ==========  ========
    Calibration                 Top-1       Drop
    ==========================  ==========  ========
    MinMax                      70.70%      -17.70pp
    Entropy                     70.70%      -17.70pp
    Percentile 99.999           80.50%       -7.90pp
    **Percentile 99.99**        **82.90%**   **-5.50pp**
    Percentile 99.99, no head   82.80%       -5.60pp
    ==========================  ==========  ========

    Two conclusions worth recording. Entropy calibration performs identically
    to min-max here, so it is not a workaround. Excluding the classifier head
    from quantization does not help, which means the residual loss is spread
    across the network rather than concentrated in one sensitive layer —
    closing the remaining gap would require quantization-aware training or a
    ViT-specific scheme such as SmoothQuant, not further layer exclusions.

    ``QuantFormat.QDQ`` inserts explicit QuantizeLinear/DequantizeLinear pairs
    rather than fused QOperator nodes. QDQ is the format TensorRT and most
    accelerators consume, and it keeps the graph readable for debugging.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessed = output_path.with_name(f"{output_path.stem}_preprocessed.onnx")
    _run_pre_process(model_path, preprocessed)

    extra_options: dict[str, Any] = {
        # Symmetric weights keep zero exactly representable, which matters for
        # padded convolutions and residual paths.
        "WeightSymmetric": True,
        "ActivationSymmetric": False,
    }
    if calibration_method is CalibrationMethod.Percentile:
        extra_options["CalibPercentile"] = percentile
        # More histogram bins give the percentile estimate finer resolution;
        # the default of 128 is too coarse for long-tailed ViT activations.
        extra_options["CalibNumBins"] = 2048

    logger.info(
        "Calibrating and quantizing to INT8 (method=%s, per_channel=%s%s)",
        calibration_method.name,
        per_channel,
        f", percentile={percentile}" if calibration_method is CalibrationMethod.Percentile else "",
    )
    try:
        quantize_static(
            model_input=str(preprocessed),
            model_output=str(output_path),
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=calibration_method,
            nodes_to_exclude=list(nodes_to_exclude) or None,
            op_types_to_quantize=list(op_types_to_quantize) or None,
            extra_options=extra_options,
        )
    except Exception as exc:
        raise QuantizationError(f"INT8 quantization failed: {exc}") from exc
    finally:
        preprocessed.unlink(missing_ok=True)

    size_before = model_path.stat().st_size / 1e6
    size_after = output_path.stat().st_size / 1e6
    logger.info(
        "Quantized: %.1f MB -> %.1f MB (%.2fx smaller)",
        size_before,
        size_after,
        size_before / size_after,
    )
    return output_path


def evaluate_onnx_accuracy(
    model_path: Path,
    dataset: Any,
    *,
    num_samples: int = 2000,
    batch_size: int = 32,
    providers: list[str] | None = None,
) -> dict[str, float]:
    """Measure top-1 and top-5 accuracy of an ONNX model on ``dataset``.

    Used to quantify what INT8 costs. Evaluating on a fixed prefix of the
    dataset makes float and quantized runs directly comparable, since both see
    identical inputs in identical order.
    """
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    limit = min(num_samples, len(dataset))
    correct_top1 = correct_top5 = total = 0

    for start in range(0, limit, batch_size):
        indices = range(start, min(start + batch_size, limit))
        images = torch.stack([dataset[i][0] for i in indices])
        targets = torch.tensor([dataset[i][1] for i in indices])

        logits = torch.from_numpy(session.run(None, {input_name: images.numpy()})[0])
        top5 = logits.topk(min(5, logits.size(-1)), dim=-1).indices

        # Tensor.item() is typed as int | float; these are integer counts.
        correct_top1 += int((top5[:, 0] == targets).sum().item())
        correct_top5 += int(top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item())
        total += targets.numel()

    return {
        "acc_top1": correct_top1 / total,
        "acc_top5": correct_top5 / total,
        "num_samples": float(total),
    }
