# Documentation Index

| Document | Read it for |
| --- | --- |
| [README](../README.md) | Setup, running the stack, API usage, headline results |
| [Technical write-up](technical-writeup.md) | Design decisions, measured results, bugs found, what is missing |
| [Model card: classifier](model-card-classifier.md) | Tiny-ImageNet ViT-Small — metrics, calibration, limitations |
| [Model card: detector](model-card-detector.md) | RT-DETR — measured mAP, licensing rationale, failure mode |
| [Model card: embedder](model-card-embedder.md) | Similarity search — retrieval quality, a published correction |
| [Benchmark data](../benchmarks/README.md) | Generated latency matrix across models and backends |
| [Benchmark analysis](../benchmarks/ANALYSIS.md) | Interpretation and the deployment recommendation |
| [OpenAPI spec](openapi.json) | Exported schema; also live at `/openapi.json` |

---

## Results at a glance

Every figure below is measured in this repository and reproducible with the
command beside it. Nothing is quoted from a publication.

### Models

| Model | Metric | Value | Reproduce |
| --- | --- | ---: | --- |
| Classifier | Top-1 (10,000 val images) | **85.88%** | `python -m models.training.train` |
| Classifier | Top-5 | 96.54% | |
| Classifier | Per-class spread | 56–100%, sd 8.6pp | `scripts/evaluate_models.py` |
| Classifier | Calibration (ECE) | 0.0852, underconfident | |
| Detector | mAP@[.5:.95] (COCO val2017) | **0.500** | `scripts/evaluate_detector.py` |
| Detector | mAP small / large | 0.347 / 0.627 | |
| Embedder | Precision@5 (1,000 queries, 200 classes) | **79.7%** | `scripts/evaluate_models.py` |
| Embedder | Per-class spread | sd 19.2pp | |

### Inference performance

Batch 32, RTX 5000 Ada. Full matrix in [`benchmarks/README.md`](../benchmarks/README.md).

| Backend | Latency | Throughput | Speedup |
| --- | ---: | ---: | ---: |
| PyTorch eager FP32 | 16.78 ms | 1,907 img/s | 1.00× |
| PyTorch eager bf16 | 5.23 ms | 6,118 img/s | 3.21× |
| **TensorRT FP16** | **3.49 ms** | **9,177 img/s** | **4.81×** |
| ONNX Runtime CPU INT8 | 475 ms | 67 img/s | 0.04× |

INT8 costs 5.4 points of accuracy and is **not recommended**; see the analysis.

### Engineering

| | |
| --- | --- |
| Tests | 368 (91.08% coverage), plus 9 performance tests |
| Serving image | 700 MB, non-root, no torch |
| CI jobs | 6 — lint, test, equivalence, model quality, security, docker |
| Alert rules | 10, all validated with `promtool` |
| Compose stacks | dev, prod, gpu — all render |
| Kubernetes | 10 resources, validated with `kubeconform --strict` |

---

## Things this project got wrong, and how

The most useful section for a reader. Every one of these was silent — the code
ran, the tests passed, nothing raised.

| Defect | How it was found | Why it mattered |
| --- | --- | --- |
| `v2.Resize` defaults to BILINEAR, not BICUBIC | Asserting numerical equality against the training transform | Shifted activations by up to 2.5 in normalised units |
| `CenterCrop` rounds, does not floor-divide | Same test | Every crop off by one pixel |
| CUDA session created successfully, failed every inference | Running the API for real | Service would start, report healthy, return 503 for all traffic |
| Every metric labelled `unmatched` | Reading detection logs | Entire Grafana dashboard useless; metrics looked fine |
| `fakeredis` without Lua | First run of the rate-limit tests | **Every rate-limit test passed vacuously** |
| Retention regex matched nothing | Testing the migration against real Postgres | Retention would silently never run, forever |
| Alert on a metric nothing exported | Writing the invariant test | Permanently green; looked like coverage |
| CI skipped 33 tests including the equivalence guard | Cloning the repo fresh and running it | The project's most important test never ran in CI |
| Cherry-picked 92% retrieval precision | Measuring all 200 classes instead of 5 | Real figure is 79.7% |

Seven of these share one shape: **code that succeeds while doing nothing.**
That is the hardest class to notice, because every signal says healthy. Two of
them are now prevented structurally by
[`tests/unit/test_invariants.py`](../tests/unit/test_invariants.py), which
fails the build if a GPU session is constructed directly or an alert references
an unexported metric — and both guards were themselves verified by introducing
a violation and confirming they failed.

The last entry is the one worth volunteering in conversation: the wrong number
was mine, published in my own model card, and the correction is recorded in
that card rather than quietly edited.

---

## What is not done

Stated plainly in [technical write-up §12](technical-writeup.md#12-what-is-still-missing):

- **Detector accuracy off-distribution** is unmeasured — mAP needs annotations
  no other dataset here provides. Its failure *mode* is characterised: it
  abstains rather than hallucinating.
- **Alert delivery endpoints** are deployment-specific. A webhook URL in a
  repository is a committed credential; the routing around it is configured
  and validated.
- **Kubernetes manifests have not been applied to a live cluster.** They
  validate against real 1.30 API schemas, which is a smaller claim than
  "deployed and working".
