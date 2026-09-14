# Model Card: Tiny-ImageNet Classifier

## Overview

| | |
| --- | --- |
| **Name** | `tiny-imagenet-classifier` |
| **Version** | v1 |
| **Task** | Image classification, 200 classes |
| **Architecture** | ViT-Small/16 (`vit_small_patch16_224.augreg_in21k_ft_in1k`) |
| **Parameters** | 21.7M |
| **Input** | RGB image, resized to 224×224 |
| **Output** | 200 logits; softmax probabilities returned ranked |
| **Licence** | Apache-2.0 (timm weights) |
| **Trained** | September 2026, single RTX 5000 Ada, 7 min 4 s |

## Intended use

Classifying photographs into the 200 Tiny-ImageNet categories — everyday
objects, animals, and scenes. Built as a demonstration of a production serving
pipeline, not as a general-purpose image classifier.

**Appropriate:** content tagging within the 200 known categories; a worked
example of an MLOps pipeline; a latency and throughput baseline for comparing
inference backends.

**Not appropriate:** any decision affecting a person's rights, safety, access to
services, or finances. Not for medical, legal, security, or surveillance use.
Not for identifying individuals — the model has no concept of identity. Not for
images unlike its training data (see limitations).

## Performance

Measured on the full 10,000-image Tiny-ImageNet validation split.

| Metric | Value |
| --- | --- |
| Top-1 accuracy | **85.88%** |
| Top-5 accuracy | **96.54%** |
| Validation loss | 0.5965 |

| Epoch | Train loss | Val loss | Top-1 | Top-5 |
| --- | --- | --- | --- | --- |
| 0 | 3.222 | 0.738 | 82.37% | 95.56% |
| 1 | 2.437 | 0.664 | 84.17% | 96.18% |
| 2 | 2.315 | 0.619 | 85.28% | 96.50% |
| 3 | 2.213 | 0.603 | 85.80% | 96.54% |
| 4 | 2.109 | 0.596 | **85.88%** | **96.54%** |

Accuracy improved monotonically with no divergence.

**Training loss is higher than validation loss, and that is expected.** Training
loss is measured against MixUp/CutMix-mixed soft targets with label smoothing,
which carry irreducible entropy — a perfectly calibrated model cannot drive it
to zero. Validation loss is measured against clean labels. The gap is evidence
the regularisers are active, not of a bug.

### Per-class accuracy

Measured over the full validation split. The aggregate hides real variation.

| | |
| --- | --- |
| Mean per-class accuracy | 85.8% |
| Standard deviation | 8.6 percentage points |
| Range | 56% – 100% |
| **Classes below 50%** | **0** |

| Weakest classes | Accuracy |
| --- | ---: |
| umbrella | 56.0% |
| pole | 58.0% |
| syringe | 58.0% |
| Egyptian cat | 60.0% |
| plate | 64.0% |

No class collapses, which is the important property — a 200-class model can
average 86% while being useless for a handful of categories. `Egyptian cat` is
instructive: it is confused with `tabby`, a genuinely fine-grained distinction
rather than a failure of the model.

### Calibration

| Metric | Value |
| --- | --- |
| Expected Calibration Error | **0.0852** |
| Mean confidence | 0.773 |
| Mean accuracy | 0.858 |
| Direction | **Underconfident by 0.085** |

The model is **systematically underconfident**: it reports 77.3% average
confidence while being right 85.8% of the time, and the gap is positive in
every single confidence bin.

| Confidence bin | n | Mean confidence | Actual accuracy | Gap |
| --- | ---: | ---: | ---: | ---: |
| 0.5–0.6 | 590 | 0.549 | 0.673 | +0.124 |
| 0.6–0.7 | 678 | 0.654 | 0.791 | +0.136 |
| 0.7–0.8 | 961 | 0.755 | 0.898 | +0.143 |
| 0.8–0.9 | 1,989 | 0.860 | 0.963 | +0.103 |
| 0.9–1.0 | 4,236 | 0.944 | 0.993 | +0.049 |

This is the expected consequence of label smoothing (0.1) and MixUp, which
deliberately prevent the model from placing full probability mass on one class.

**Underconfidence is the safe direction**, but it is not harmless: a caller
thresholding at 0.9 to filter for "high confidence" predictions discards a
large number of results that are correct 96% of the time. A caller wanting a
95% precision operating point should threshold around **0.75**, not 0.95.

### Serving-path accuracy

The production API reimplements preprocessing in NumPy so the serving container
needs no torch, which means two independent implementations must agree.

Measured on the **full 10,000-image validation split**, passed as raw bytes
through the serving preprocessing and the exported ONNX graph: **85.78%
top-1**, against 85.88% from the training loop. The 0.10pp difference is
attributable to floating-point ordering between the two preprocessing
implementations, which agree to 7.2e-07 per pixel.

That agreement is asserted by
`tests/unit/test_image_processing.py::TestTorchvisionEquivalence` across eight
input shapes, and is the only thing preventing the two implementations from
drifting apart — see the write-up's discussion of the two silent preprocessing
bugs found that way.

### Latency

Batch 32, RTX 5000 Ada. Full matrix in [`benchmarks/README.md`](../benchmarks/README.md).

| Backend | Latency | Throughput | Top-1 |
| --- | ---: | ---: | ---: |
| PyTorch eager FP32 | 16.78 ms | 1,907 img/s | 87.25%* |
| PyTorch eager bf16 | 5.23 ms | 6,118 img/s | 87.25%* |
| TensorRT FP16 | 3.49 ms | 9,177 img/s | 87.25%* |
| ONNX Runtime CPU INT8 | 475 ms | 67 img/s | 81.85%* |

\* measured on a fixed 2,000-image subset, so slightly above the full-split figure.

Single-image latency is 1.05 ms under TensorRT.

## Training

**Data.** Tiny-ImageNet: 100,000 training and 10,000 validation images at 64×64,
across 200 classes, roughly 500 training images per class.

**Procedure.** Five epochs of fine-tuning with AdamW, batch size 128, bf16 mixed
precision, gradient clipping at 1.0, cosine schedule with 5% warmup, and
layer-wise learning-rate decay (head 1e-3, encoder 1e-4, decay 0.75 across 28
parameter groups).

**Augmentation.** RandomResizedCrop with scale (0.65, 1.0) — far above the
ImageNet default of 0.08, because at 64×64 an 8% area crop frequently contains
none of the labelled object — horizontal flip, RandAugment (2 ops, magnitude 9),
PCA lighting jitter, random erasing at p=0.25, and MixUp/CutMix with label
smoothing 0.1.

## Limitations

**The headline number is less impressive than it looks.** Tiny-ImageNet's 200
classes are a *subset of ImageNet-1k*, and the backbone was pretrained on
ImageNet-21k then fine-tuned on ImageNet-1k. It had already seen every target
category before fine-tuning began. This is a property of the transfer setup, not
evidence of a strong method, and the same recipe on genuinely novel categories
would perform considerably worse.

**Native resolution is 64×64, upsampled to 224.** No detail is recovered by
upsampling. The model performs worse on small objects and fine texture than the
224×224 figures might suggest.

**Closed-set.** Every input is assigned one of the 200 classes. An image of
something outside the label space — a car, a building, a document — receives a
confident, wrong prediction. There is no abstention mechanism and no
out-of-distribution detection. **This is the most likely way the model fails in
practice**, and any caller should treat predictions as valid only for inputs
known to belong to the 200 categories.

**Probabilities are systematically underconfident** (ECE 0.085). They are
usable for ranking, and conservative as a confidence signal, but a caller
thresholding at a nominal probability gets better precision than the number
suggests. See the calibration table above for the correct operating points.

**Per-class performance varies by 8.6 percentage points** (56%–100%). No class
falls below 50%, but a caller relying on the aggregate for a specific weak
category — `umbrella`, `pole`, `syringe` — will be disappointed. See the
per-class table above.

**Robustness is untested.** No evaluation against corruption, compression
artefacts, adversarial perturbation, or distribution shift.

## Ethical considerations

**Inherited bias.** The model inherits whatever biases exist in ImageNet, which
are well documented: geographic skew toward North America and Europe, and
historical problems with person-related categories. Tiny-ImageNet contains
person-adjacent classes. Predictions on images of people should be treated as
unreliable and should not inform any decision about a person.

**No consent provenance.** ImageNet and Tiny-ImageNet were assembled from web
images without subject consent. Deployments in jurisdictions with biometric or
image-rights regimes should confirm their legal position independently.

**Audit trail.** Every inference is recorded with caller, model version,
timestamp, and prediction, so a disputed output can be traced to a specific
artefact. No image bytes are retained — only a SHA-256 digest. Operators must
set a retention policy; nothing here expires records automatically.

**Dual use.** The pipeline generalises to any classification task, including
harmful ones. The training code is task-agnostic; responsibility sits with the
operator.

## Reproducing

```bash
python scripts/setup/download_datasets.py --dataset tiny_imagenet
uv run python -m models.training.train
uv run python scripts/prepare_artifacts.py
```

Seed 42, `deterministic_cuda=false` — run-to-run variance is well below the
differences discussed here, and the fast path is the one production uses.

## Maintenance

No retraining schedule; this is a demonstration artefact. In production the
drift signals worth monitoring would be the input distribution (image
dimensions, formats, submission rate per class) and the prediction distribution,
both available from the `inference_logs` table. A sustained shift in predicted
class frequencies is the earliest practical warning that inputs have moved away
from the training distribution.
