# Model Card: Image Similarity Embedder

## Overview

| | |
| --- | --- |
| **Name** | `tiny-imagenet-embedder` |
| **Version** | v1 |
| **Task** | Image similarity search |
| **Architecture** | ViT-Small/16, supervised-contrastive fine-tune (`train_embedder`) |
| **Embedding** | 384-d, L2-normalised |
| **Index** | FAISS `IndexFlatIP` over 20,000 training images |
| **Metric** | Cosine similarity |
| **Licence** | Apache-2.0 (timm weights) |

## Design

The embedder is a **dedicated** ViT-Small trained with supervised contrastive
loss (`config/train_embedder.yaml`). Classification features collapse
intra-class variation by construction; a contrastive objective pulls same-class
images together without requiring that collapse.

An older classification checkpoint can still be exported as an embedder (the
head is dropped at load time) so the original artefact layout keeps working.
New indexes should come from the contrastive run.

Consequently this retrieves **semantically** similar images (same category,
comparable pose and colour), not visually near-duplicate ones. For "show me more
like this" that is usually the intent. For near-duplicate or copyright
detection it is the wrong tool, and a perceptual hash would be correct.

**Why cosine.** Embeddings are L2-normalised — normalisation is baked into the
exported ONNX graph so serving code cannot omit it — and inner product on unit
vectors is exactly cosine similarity. Raw L2 distance on un-normalised
transformer features is dominated by vector magnitude, which tracks image
contrast rather than content.

**Why an exact index.** `IndexFlatIP` performs brute-force search. At 20,000
vectors that takes under a millisecond. An approximate index (IVF, HNSW) would
be faster above roughly a million vectors but adds a training step and costs
recall; choosing approximation before exact search is too slow is premature.

## Performance

Measured over **1,000 validation queries sampled at random from the validation
split**, which covered 198 of the 200 classes
against the 20,000-image training index. Queries come from validation and the
index from training, so a query can never retrieve itself.

| Metric | Value |
| --- | --- |
| Precision@1 | **78.3%** |
| **Precision@5** | **77.0%** |
| Precision@10 | 75.9% |
| Query latency | ~5 ms (ONNX Runtime, including embedding) |
| Index size | 30 MB, 20,000 vectors |
| Export fidelity | max abs. difference **2.682e-07** vs PyTorch ([measured](../benchmarks/export_fidelity.json)) |

> **A correction.** An earlier version of this card reported 92% precision@5.
> That figure came from five hand-picked classes and did not survive proper
> measurement: the classification-backbone index scored 79.7%, and the
> contrastive index that is actually served scores **77.0%** across 1,000
> random queries. The original sample happened to contain visually distinctive
> categories. It is recorded here because a cherry-picked benchmark that
> flatters the model is exactly the kind of number that should not be trusted,
> including when it is your own.

### Per-class variation is large

Mean per-class P@5 is 77.6% with a standard deviation of **20.3 percentage
points** — far wider than the classifier's 8.6pp. Retrieval is much less
uniform than classification.

| Weakest classes | P@5 |
| --- | ---: |
| bannister | 0.0% |
| pole | 0.0% |
| rocking chair | 20.0% |
| water jug | 25.7% |
| stopwatch | 32.0% |

The pattern is consistent: categories defined by *context* rather than
appearance (a pole, a bannister) retrieve at zero, because the embedding
captures overall scene composition and those objects rarely dominate their
frame. Categories with distinctive colour and texture retrieve near-perfectly.

**A caller should not assume uniform quality.** For a context-dependent
category, retrieval is close to useless.

### INT8 is built, measured, and not deployed

Top-1 accuracy does not apply to a model with no classifier head, so the
question is whether quantized vectors still rank neighbours the way FP32 ones
do. They do not.

| Metric | INT8 vs FP32 |
| --- | ---: |
| Mean cosine similarity | 0.717 |
| recall@5 searching the FP32 index | **0.505** |
| Artefact size | 87.8 -> 23.2 MB |

Both numbers are needed. High cosine similarity alone would not be sufficient —
a uniform rotation would preserve it while destroying retrieval — and recall@5
measures the mixed FP32-index/INT8-query regime the service would actually run
in. Rebuilding the index in INT8 would not rescue it: at 0.717 mean similarity
the quantization noise is comparable to the distance between genuine
neighbours.

Latency does not justify it either. INT8 saves 1.1 ms at batch 1 on CPU
(12.3 ms vs 13.4 ms), against 1.06 ms for TensorRT FP16 on GPU.

Measured by `evaluate_embedding_fidelity`; raw numbers in
`benchmarks/quantization.json`.

## Index construction

Built from the **training** split, never validation. Indexing validation images
would make any evaluation circular — a validation query would retrieve itself
and report perfect precision.

100 images are sampled from each of the 200 classes. Sampling evenly matters:
taking the first 20,000 files alphabetically would index only the first few
dozen categories and make retrieval useless for everything else.

```bash
uv run python scripts/prepare_artifacts.py --with-similarity
uv run python scripts/build_similarity_index.py --num-images 20000
```

## Limitations

**Retrieval is limited to the indexed set.** Only the 20,000 indexed training
images can be returned. A query unlike anything indexed still returns its five
nearest neighbours, with low similarity scores as the only signal that the
result is poor. There is no "no good match" response.

**Closed domain.** The backbone was fine-tuned on 200 Tiny-ImageNet categories.
Similarity between images outside that domain is unreliable — the features were
never optimised to separate them.

**Native 64×64 source imagery.** Indexed images are upsampled from 64×64. Fine
texture and small detail are absent from the embeddings, so similarity is driven
by coarse shape and colour.

**Per-class retrieval varies by 20.3 percentage points**, more than twice the
classifier's spread. The 77.0% aggregate is a poor predictor for any specific
category: context-defined ones (`pole`, `bannister`) retrieve at 0%. A
caller should not assume uniform quality.

**Memory-resident.** The full index loads into memory at startup. At 30 MB that
is trivial; at 10 million vectors (roughly 15 GB) it would require an on-disk or
sharded index.

## Ethical considerations

**Not for identification.** Retrieving similar images must not be used to
identify people, locations, or property. The model has no concept of identity,
and similarity is not evidence of any relationship between two images.

**Retrieval can leak training data.** A similarity search returns *references
to indexed images*. Where an index is built over private or user-uploaded
content, search becomes a channel for exposing that content to whoever can
query it. Only public training images are indexed here; any deployment over
user data needs access control at the index level, not just at the API.

**Inherited bias.** The backbone's ImageNet biases carry into the embedding
space. Categories under-represented in pretraining will retrieve less reliably,
and that variation is not characterised.

## Upstream

Same weights as the classifier — see
[`model-card-classifier.md`](model-card-classifier.md) for training procedure,
data provenance, and the shared ImageNet limitations.
