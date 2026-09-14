# Model Card: Image Similarity Embedder

## Overview

| | |
| --- | --- |
| **Name** | `tiny-imagenet-embedder` |
| **Version** | v1 |
| **Task** | Image similarity search |
| **Architecture** | ViT-Small/16 backbone, classifier head removed |
| **Embedding** | 384-d, L2-normalised |
| **Index** | FAISS `IndexFlatIP` over 20,000 training images |
| **Metric** | Cosine similarity |
| **Licence** | Apache-2.0 (timm weights) |

## Design

The embedder **is** the fine-tuned classifier with its head removed. That is a
deliberate reuse with a real trade-off.

**Why reuse.** The backbone was fine-tuned on this exact domain, so its
penultimate features are already discriminative for these categories — which is
what similarity search needs. It costs one extra 88 MB export and no additional
training. A separate CLIP model would give better open-domain similarity but
would add a third set of weights, a second preprocessing pipeline, and a model
that has never seen this data.

**The trade-off, stated plainly.** Features optimised for classification
*collapse intra-class variation by construction* — that is precisely what makes
a classifier work. Two different goldfish photographs are near-identical in this
space.

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

| Metric | Value |
| --- | --- |
| **Precision@5** | **92%** (23/25 retrieved images share the query's class) |
| Query latency | ~5 ms (ONNX Runtime CPU, including embedding) |
| Index size | 30 MB, 20,000 vectors |
| Export fidelity | max abs. difference 2.7e-07 vs PyTorch |

Measured across five query classes drawn from the validation split. Retrieval
quality varies noticeably by class: visually distinctive categories (goldfish,
tabby, orange) returned 5/5 with similarities of 0.72–0.84, while an ambiguous
category (`reel`) returned 3/5 at 0.45–0.47.

**Lower similarity scores indicate lower confidence**, and the `reel` result
shows the pattern clearly. A caller can use `min_similarity` to trade recall for
precision, though no calibrated threshold has been established.

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

**Precision@5 was measured on five classes**, not all 200. The 92% figure is
indicative, not a rigorous benchmark, and per-class variation is demonstrably
large.

**No index freshness mechanism.** The index is a static artefact. Adding images
requires a rebuild; there is no incremental update path.

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
