# Model Card: RT-DETR Object Detector

## Overview

| | |
| --- | --- |
| **Name** | `rtdetr-coco-detector` |
| **Version** | v1 |
| **Task** | Object detection, 80 COCO classes |
| **Architecture** | RT-DETR R18 (`PekingU/rtdetr_r18vd_coco_o365`) |
| **Parameters** | 20.2M |
| **Input** | RGB image, resized to 640×640 |
| **Output** | 300 queries × (80 class scores, 4 box coordinates) |
| **Licence** | **Apache-2.0** (implementation and weights) |
| **Provenance** | Used pretrained; not fine-tuned here |

## Intended use

Detecting and localising the 80 COCO object categories in photographs.
Integrated to demonstrate multi-model serving; the weights are used as
published.

**Appropriate:** general object detection within the COCO categories; a second
workload for exercising the serving stack's multi-model paths.

**Not appropriate:** safety-critical detection (autonomous driving, industrial
safety, medical imaging). Not for surveillance or person-tracking. Not for
counting where an undercount has consequences — recall is unmeasured on any
distribution other than COCO.

## Why this model

**Licensing was the deciding factor.** Ultralytics' YOLOv8 and YOLOv11 are
**AGPL-3.0**, which obliges anyone offering the software over a network to make
their entire source available. For a commercial product that is usually
disqualifying, and it is an expensive decision to reverse once the model is
embedded. RT-DETR is Apache-2.0 in both its implementation and its published
weights. The challenge names DETR explicitly.

**R18 rather than R101.** 20M parameters against 76M, and roughly a quarter of
the latency. The *published* COCO mAP for the two backbones is 46.5 against
54.3 — those are the figures the selection was made on, and are distinct from
the 0.500 measured here (see Performance below, which uses the
Objects365-pretrained checkpoint on a 1,000-image subset). For a demonstration
API where classification is the primary workload, the accuracy difference is
not worth 4× the compute.

**It exports cleanly.** RT-DETR is anchor-free and NMS-free: it predicts a fixed
set of 300 queries with a one-to-one assignment loss, so duplicates are
suppressed during training rather than by a post-processing step. NMS is
data-dependent and exports poorly to ONNX, and is the usual source of pain when
deploying detectors.

## Performance

**Measured** on COCO val2017 with `pycocotools`, the reference implementation
used by the COCO leaderboard — not quoted from the model's publication.

| Metric | Measured |
| --- | ---: |
| **mAP@[.5:.95]** | **0.500** |
| mAP@0.5 | 0.667 |
| mAP@0.75 | 0.539 |
| mAP small | 0.347 |
| mAP medium | 0.516 |
| mAP large | 0.627 |
| AR@100 | 0.686 |

Evaluated on 1000 images at a score threshold of
0.01. The threshold is deliberately far below the serving
default of 0.5: mAP integrates precision over the full recall curve, so
discarding low-scoring detections truncates the curve and *understates* the
score. 0.5 is an operating point for a user; it is the wrong threshold for
measuring a model.

**Small objects are measurably the weakness**: 0.347 against
0.627 for large objects — a gap of
0.280. This is characteristic of the model
family, and the R18 backbone is weaker here than deeper variants. A caller
detecting small or distant objects should expect materially worse results than
the headline figure.

### Latency

| Backend | Batch 1 | Batch 8 |
| --- | ---: | ---: |
| ONNX Runtime CPU FP32 | 54.1 ms | 537.4 ms |
| ONNX Runtime CPU INT8 | 62.9 ms | 496.9 ms |
| ONNX Runtime CUDA | 3.7 ms | 29.2 ms |
| **TensorRT FP16** | **2.1 ms** | **8.3 ms** |

TensorRT gives 1.7x over the CUDA provider. Batches above 8 are not benchmarked:
at 640x640 a detection input is roughly eight times the pixels of a 224x224
classification input, so large batches exhaust GPU memory long before they
saturate compute.

Export fidelity: max absolute difference **3.610e-06** against PyTorch,
re-measured against the committed artefact by `scripts/verify_exports.py` and
recorded in [`benchmarks/export_fidelity.json`](../benchmarks/export_fidelity.json).
Artefact size 84.3 MB.

### INT8 is built, measured, and not deployed

| Metric | FP32 | INT8 |
| --- | ---: | ---: |
| mAP@[.5:.95] | 0.4999 | **0.0632** |
| mAP@0.5 | 0.6674 | 0.0880 |
| mAP small | 0.3469 | **0.0000** |
| mAP medium | 0.5163 | 0.0240 |
| mAP large | 0.6271 | 0.1490 |
| Artefact size | 84.3 MB | 24.6 MB |

Quantization destroys this model: 87% of its mAP, and **no small object is
detected at all**. Detection is dominated by box regression, which has none of
the margin a classifier's argmax enjoys — quantization noise is roughly
constant in absolute pixels, so it is proportionally fatal for the smallest
boxes and merely severe for the largest. That monotonic small-to-large gradient
is the signature of coordinate noise rather than bad calibration.

It also fails *confidently*: the quantized model still emits ~246k detections
against FP32's ~250k, so nothing downstream can tell that the boxes are wrong.
INT8 is additionally **slower** than FP32 at batch 1 (62.9 ms vs 54.1 ms).

Measured with `scripts/evaluate_detector.py --model detector_int8.onnx`; raw
numbers in `benchmarks/detection_eval_int8.json`.

Functional verification on COCO `000000039769` (two cats on a couch with two
remotes). Six detections above 0.5, at 0.763–0.950 confidence, every box inside
the image bounds:

| Score | Label | Box |
| ---: | --- | --- |
| 0.950 | cat | (344, 25, 640, 372) |
| 0.949 | cat | (11, 56, 316, 472) |
| 0.925 | remote | (41, 73, 175, 117) |
| 0.860 | remote | (334, 76, 371, 188) |
| 0.850 | sofa | (0, 0, 640, 479) |
| 0.763 | sofa | (0, 0, 640, 479) |

Six detections, but **not** the six ground-truth objects: the annotation lists
`cat, cat, remote, remote, couch, bed`, and the model returns the couch twice
while missing the bed entirely. Worth recording for two reasons. The duplicate
is a near-exact repeat of the same whole-image box, which is mildly surprising
for an NMS-free architecture whose one-to-one assignment loss is supposed to
suppress exactly this. And the matching count is a coincidence — reading "six
detections" as "six objects found" is the kind of eyeball verification that
looks like evidence and is not. The measured claim is the mAP above.

## Preprocessing

RT-DETR's preprocessing differs from the classifier's in two ways that are
silent when wrong, and both are asserted in tests:

1. **No mean/std normalisation.** Its processor sets `do_normalize=False`;
   inputs are only rescaled to `[0, 1]`. Applying ImageNet statistics by
   analogy with the classifier shifts the input distribution and degrades
   detection with no error raised.
2. **Square resize without preserving aspect ratio.** The processor resizes
   directly to 640×640 with `do_pad=False`. Letterboxing instead would place
   objects where the model does not expect them, and would make the inverse box
   transform wrong.

## Post-processing

**Per-class sigmoid, not softmax.** Each query is scored against every class
independently, because a region may legitimately match several labels. Softmax
across classes would force scores to sum to one and systematically suppress
confident multi-label detections. The sigmoid is baked into the exported graph
so a reimplementation cannot get it wrong.

**No NMS.** Adding it would remove legitimate detections of genuinely
overlapping objects.

**One detection per query.** Each query contributes only its highest-scoring
class; emitting every class above threshold would report one object several
times under different labels.

Boxes are returned in absolute pixel coordinates of the uploaded image and
clipped to its bounds, so a client can overlay them directly.

## Limitations

**Evaluated on COCO only.** mAP of 0.500 is measured on COCO val2017, the
distribution the model was trained for. Accuracy elsewhere is unmeasured,
because mAP needs annotations no other dataset here provides.

Its *failure mode* off-distribution has, however, been characterised. Probed on
Tiny-ImageNet (64x64 upscaled — same object categories, entirely unlike COCO
imagery):

| | COCO | Tiny-ImageNet |
| --- | ---: | ---: |
| Mean peak confidence | 0.899 | 0.430 |
| Mean detections at 0.5 | 5.44 | 0.33 |
| Images with no detection | 1/150 | 106/150 |

Confidence falls 52% and detections
94%; 106 of
150 images return nothing at all.

**The model abstains rather than hallucinating, which is the recoverable
failure mode.** A caller receiving empty results can detect the condition and
escalate; a model returning confident wrong boxes gives nothing downstream any
way to tell. This does not make it safe to use off-distribution — it means
that when it is misused, the misuse is visible.

Reproduce with `scripts/evaluate_detector.py --probe-off-distribution`.

**Fixed 640×640 input.** The model has learned positional priors at this
resolution; it is not a tunable parameter. Images far from square are squashed,
which degrades detection of elongated objects.

**Small objects are measurably weaker**: mAP 0.347 against
0.627 for large objects. See the performance table.

**Closed vocabulary.** Only the 80 COCO classes. Anything else is either missed
or misassigned to the nearest category.

**Confidence threshold is caller-supplied and consequential.** The default of
0.5 is a convention, not a calibrated operating point. No precision/recall curve
was computed, so a caller choosing a threshold is doing so blind.

**Cold-start latency.** The first request after startup pays initialisation. The
model service runs a warmup inference at load time so this cost is paid during
startup rather than by a user.

## Ethical considerations

**Person detection.** `person` is one of the 80 classes. This model must not be
used for surveillance, tracking, or any identification purpose. It detects that
a person is present; it has no concept of identity, and should not be built into
a system that infers one.

**COCO's biases are inherited.** COCO is geographically and contextually skewed
toward North American and European scenes. Detection quality varies with
context, clothing, and setting in ways that are not characterised here.

**Failure asymmetry.** For most applications a missed detection and a false
detection have very different costs. No analysis of that trade-off was done, and
the default threshold does not encode one.

## Reproducing

```bash
uv run python scripts/prepare_artifacts.py --with-detection
```

Downloads the pretrained checkpoint, exports to ONNX, and verifies numerical
equivalence against the PyTorch reference before writing the artefact.

## Upstream

- Model: [`PekingU/rtdetr_r18vd_coco_o365`](https://huggingface.co/PekingU/rtdetr_r18vd_coco_o365)
- Paper: *DETRs Beat YOLOs on Real-time Object Detection* (Zhao et al., 2023)
- Licence: Apache-2.0
