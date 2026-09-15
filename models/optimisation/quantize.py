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
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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


class ImageFileCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed image files to the calibrator.

    The detector has no torch ``Dataset`` behind it — it consumes COCO JPEGs
    through the same ``preprocess_for_detection`` the API calls on every
    request. Calibrating through the serving code path rather than a
    reimplementation is the point: RT-DETR skips mean/std normalisation and
    resizes without preserving aspect ratio, and calibrating on differently
    preprocessed inputs would fit activation ranges to a distribution the model
    never sees in production.

    ``preprocess`` takes the raw file bytes and returns a CHW float32 array.
    """

    def __init__(
        self,
        paths: list[Path],
        input_name: str,
        preprocess: Callable[[bytes], np.ndarray],
        *,
        num_samples: int = DEFAULT_CALIBRATION_SAMPLES,
        batch_size: int = 4,
    ) -> None:
        self.paths = paths[:num_samples]
        self.input_name = input_name
        self.preprocess = preprocess
        self.batch_size = batch_size
        self._iterator: Iterator[dict[str, np.ndarray]] | None = None

    def _batches(self) -> Iterator[dict[str, np.ndarray]]:
        buffer: list[np.ndarray] = []
        for path in self.paths:
            buffer.append(self.preprocess(path.read_bytes()))
            if len(buffer) == self.batch_size:
                yield {self.input_name: np.stack(buffer)}
                buffer = []

        if buffer:
            yield {self.input_name: np.stack(buffer)}

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


#: Temporary files ONNX Runtime's preprocessing writes into the *current*
#: directory and never removes. `sym_shape_infer_temp.onnx` is hardcoded in the
#: shape-inference pass; the UUID-named `.data` files are external-tensor
#: sidecars, one per model too large for a single protobuf.
_STRAY_TEMP_PATTERNS = ("sym_shape_infer_temp.onnx", "*.data")


@contextmanager
def _clean_workspace() -> Iterator[Path]:
    """Yield a scratch directory and remove what the toolchain leaves behind.

    Two kinds of litter have to be handled differently. The preprocessed model
    and its external-data sidecar go in a temporary directory, which disappears
    wholesale. `sym_shape_infer_temp.onnx` and the sidecars ONNX Runtime writes
    for its *own* intermediates land in the working directory instead, so those
    are swept explicitly.

    The sweep only removes files that were not present beforehand, so a
    legitimately named `.data` file in the working directory survives. Worth
    doing: each sidecar is the size of the model, and four quantization runs
    left 325 MB of untracked files in the repository root -- one `git add .`
    away from being committed.
    """
    cwd = Path.cwd()
    before = {path.name for path in cwd.iterdir()}

    with tempfile.TemporaryDirectory(prefix="quantize-") as scratch:
        try:
            yield Path(scratch)
        finally:
            for pattern in _STRAY_TEMP_PATTERNS:
                for stray in cwd.glob(pattern):
                    if stray.name not in before:
                        stray.unlink(missing_ok=True)
                        logger.debug("Removed stray temporary file %s", stray.name)


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
    with _clean_workspace() as scratch:
        preprocessed = scratch / f"{output_path.stem}_preprocessed.onnx"
        _run_pre_process(model_path, preprocessed)

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


def evaluate_embedding_fidelity(
    fp32_path: Path,
    int8_path: Path,
    dataset: Any,
    *,
    num_samples: int = 1000,
    batch_size: int = 32,
    providers: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, float]:
    """Measure what INT8 costs an embedder.

    Top-1 accuracy is meaningless here: the embedder has no classifier head, it
    emits a 384-dimensional unit vector. What matters is whether the quantized
    vectors still rank neighbours the same way, because the FAISS index is built
    from FP32 vectors and a quantized *query* against it only works if the two
    spaces agree.

    Two numbers are reported, and both are needed:

    ``mean_cosine_similarity``
        How close each INT8 vector is to its FP32 counterpart. High similarity
        is necessary but not sufficient — a uniform rotation would preserve it
        while destroying retrieval against an un-rotated index.

    ``recall_at_k``
        The fraction of each FP32 query's true top-k neighbours that the INT8
        query still retrieves, searching the *FP32* index. This is the mixed
        regime the service would actually run in, and it is the number that
        decides whether the artefact is usable.
    """
    import onnxruntime as ort
    import torch

    def embed(model_path: Path) -> np.ndarray:
        session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        limit = min(num_samples, len(dataset))

        chunks = []
        for start in range(0, limit, batch_size):
            images = torch.stack(
                [dataset[i][0] for i in range(start, min(start + batch_size, limit))]
            )
            chunks.append(session.run(None, {input_name: images.numpy()})[0])
        return np.concatenate(chunks).astype(np.float32)

    fp32 = embed(fp32_path)
    int8 = embed(int8_path)

    if len(fp32) <= top_k:
        raise ValueError(
            f"recall@{top_k} needs more than {top_k} samples to rank against; "
            f"got {len(fp32)}. Raise --accuracy-samples."
        )

    # Re-normalise: the graph bakes in L2 normalisation, but quantization
    # perturbs the output so the INT8 vectors are only approximately unit
    # length. Comparing un-normalised vectors would conflate a magnitude shift
    # with an angular one.
    fp32 /= np.linalg.norm(fp32, axis=1, keepdims=True)
    int8 /= np.linalg.norm(int8, axis=1, keepdims=True)

    cosine = float((fp32 * int8).sum(axis=1).mean())

    # Rank against the FP32 population, excluding self-matches, and compare the
    # two neighbour sets. argpartition is O(n) per row against sort's O(n log n).
    def neighbours(queries: np.ndarray) -> np.ndarray:
        scores = queries @ fp32.T
        np.fill_diagonal(scores, -np.inf)
        top = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
        return np.sort(top, axis=1)

    fp32_neighbours = neighbours(fp32)
    int8_neighbours = neighbours(int8)

    overlap = [
        len(set(a.tolist()) & set(b.tolist()))
        for a, b in zip(fp32_neighbours, int8_neighbours, strict=True)
    ]

    return {
        "mean_cosine_similarity": cosine,
        "recall_at_k": float(np.mean(overlap) / top_k),
        "top_k": float(top_k),
        "num_samples": float(len(fp32)),
    }
