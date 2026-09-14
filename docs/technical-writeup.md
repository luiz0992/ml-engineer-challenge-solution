# Technical Write-up

Design decisions, measured results, and the reasoning behind both. Companion to
the [README](../README.md) (how to run it) and
[`benchmarks/ANALYSIS.md`](../benchmarks/ANALYSIS.md) (optimisation detail).

---

## 1. Model selection

### Classification: ViT-Small/16

Chosen because Tiny-ImageNet's 200 classes are a **subset of ImageNet-1k**, so
an ImageNet-21k→1k pretrained backbone has already seen every target category.
That makes transfer unusually effective — 82.4% top-1 after a single epoch — and
it is the honest explanation for the headline number rather than anything clever
in the method.

22M parameters keeps a full run to seven minutes on one GPU, which matters more
than the last point of accuracy for a project whose subject is the *pipeline*. A
ViT-Base would have added roughly 4× the compute for perhaps two points.

**Layer-wise learning-rate decay** was the single most valuable training choice.
The classifier head is randomly initialised while the backbone is not; a uniform
learning rate either moves the head too slowly or destroys pretrained features
with early gradients from an untrained head. Scaling the rate by `0.75^depth`
across 28 parameter groups (2.4e-06 to 1.0e-03) resolves that tension directly.

### Detection: RT-DETR R18

**Licensing decided this.** Ultralytics YOLOv8/v11 are AGPL-3.0 — offering the
software over a network obliges you to publish your entire source. That is
usually disqualifying commercially, and very expensive to unwind after the model
is embedded. RT-DETR is Apache-2.0 throughout.

It also exports cleanly: anchor-free, NMS-free, with a one-to-one assignment
loss that suppresses duplicates during training. NMS is data-dependent and
exports poorly, and is the usual source of deployment pain for detectors.

### Similarity search: the classifier backbone reused

The third model is the fine-tuned classifier with its head removed, indexed with
FAISS. The trade-off is explicit: features optimised for classification collapse
intra-class variation *by construction*, so this retrieves semantically similar
images rather than visually near-duplicate ones. For "more like this" that is
the intent; for near-duplicate detection it is the wrong tool and a perceptual
hash would be correct.

Reuse costs one extra export and no additional training. A separate CLIP model
would give better open-domain similarity at the cost of a third set of weights,
a second preprocessing pipeline, and a model that has never seen this data.

Measured precision@5 is 92% across five query classes, at ~5 ms per query over
20,000 indexed images. Quality varies sharply by class — distinctive categories
returned 5/5 at similarity 0.72-0.84, an ambiguous one returned 3/5 at 0.45.

---

## 2. Optimisation results

Batch 32, RTX 5000 Ada, eager PyTorch FP32 as baseline.

| Backend | Latency | Throughput | Speedup | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| PyTorch eager FP32 | 16.78 ms | 1,907 img/s | 1.00× | 87.25% |
| PyTorch eager bf16 | 5.23 ms | 6,118 img/s | **3.21×** | 87.25% |
| **TensorRT FP16** | **3.49 ms** | **9,177 img/s** | **4.81×** | 87.25% |
| ONNX Runtime CUDA FP32 | 18.86 ms | 1,696 img/s | 0.90× | 87.25% |
| ONNX Runtime CPU INT8 | 475 ms | 67 img/s | 0.04× | 81.85% |

### INT8 failed, and is not deployed

Static quantisation cost **5.4 points of top-1** for a 1.3× CPU speedup. Reaching
even that required replacing the default calibration:

| Calibration | Top-1 | Drop |
| --- | ---: | ---: |
| MinMax (ONNX Runtime default) | 70.70% | −17.70pp |
| Entropy | 70.70% | −17.70pp |
| Percentile 99.999 | 80.50% | −7.90pp |
| **Percentile 99.99** | **82.90%** | **−5.50pp** |
| Percentile 99.99, head excluded | 82.80% | −5.60pp |

Min-max sets the quantisation range from the single most extreme activation
observed. LayerNorm and GELU in vision transformers produce rare outliers orders
of magnitude above typical values, so one outlier stretches the range until
ordinary activations collapse into a handful of the 256 available levels.
Percentile calibration discards the top 0.01% and recovers 12 points.

Two negative results are worth recording: **entropy calibration performed
identically to min-max**, so it is not an alternative fix; and **excluding the
classifier head changed nothing**, which means the residual loss is spread
across the network rather than concentrated in one sensitive layer. Closing the
gap would require quantisation-aware training or a ViT-specific scheme such as
SmoothQuant — not more layer exclusions. Establishing that was worth more than
further PTQ tuning.

### ONNX Runtime CUDA is slower than PyTorch

0.90× — the opposite of the common expectation. PyTorch dispatches ViT attention
to a fused SDPA (FlashAttention) kernel; the exported graph decomposes attention
into separate MatMul, Div, Softmax, and Transpose nodes that the CUDA provider
executes individually, paying memory bandwidth at each step. TensorRT wins
precisely because it re-fuses them during engine building.

**ONNX export is valuable for portability and as the route to TensorRT, not as a
speedup in itself.** A team exporting and expecting a free win would be
disappointed.

### Recommendation

**Serve bf16.** 3.21×, zero accuracy change, one configuration flag, no build
step, portable. Add TensorRT where throughput justifies per-shape engine builds
and non-portable artefacts. Do not deploy INT8 for this model.

---

## 3. Architecture

```
client → nginx gateway → FastAPI → ONNX Runtime
                            ├── Redis   (cache + rate limits + broker)
                            ├── Postgres (audit trail)
                            └── Celery worker (batch)
                         Prometheus → Grafana
```

### Decisions worth defending

**The serving image contains no torch.** Preprocessing is reimplemented in NumPy
and Pillow, keeping the image at 700 MB instead of several gigabytes. The cost
is that two implementations of the same transform must agree — addressed in §4.

**Networks are segmented.** Postgres and Redis sit on a backend network only,
unreachable from anything exposed externally. The API is not published to the
host, so traffic cannot bypass the gateway's rate limiting or its `/metrics`
deny rule.

**Two rate limiters, deliberately.** nginx limits per IP and absorbs floods,
including unauthenticated ones that never reach the application. The application
limits per authenticated user and tier. Neither substitutes for the other.

**The cache never fails a request.** A Redis outage degrades to uncached
inference. Cache keys include model name, version, backend, and request options,
so a rollout cannot serve predictions produced by the previous configuration.

**Audit writes are queued, not awaited.** A synchronous insert would add a
network round trip to a request whose entire budget is a few milliseconds — the
audit trail would cost more than the inference. The queue is bounded, so a
database outage degrades auditing rather than growing memory until the process
is OOM-killed. Shutdown drains it, because otherwise audit gaps cluster exactly
around deploys.

**Liveness checks nothing.** It answers "should this container be restarted".
Making it depend on Redis or Postgres would restart healthy processes when
something downstream failed, turning a partial outage into a crash loop.
Dependency checks live in readiness.

**Health distinguishes degraded from unhealthy.** A service running on a fallback
backend, or without a cache, is still serving and must stay in the load
balancer. Only the loss of inference itself is unhealthy.

**Graceful degradation is verified, not assumed.** Every backend must complete a
warmup inference before it is accepted — see §4.

---

## 4. Four bugs worth reporting

Each was silent. Each is now covered by a test.

### Preprocessing divergence (two bugs, one test)

`torchvision.transforms.v2.Resize` defaults to **BILINEAR**, not BICUBIC. And
`CenterCrop` uses `round((dim - size) / 2)`, not floor division — for a 275-wide
image, `round(25.5) = 26` against `51 // 2 = 25`.

Neither raises. Together they shifted activations by up to 2.5 in normalised
units and moved every crop by a pixel — precisely the kind of defect that
surfaces only as accuracy inexplicably below the benchmark. Found by asserting
numerical equality against the training transform across eight input shapes; now
agreeing to 7.2e-07. Since the serving path deliberately reimplements
preprocessing, that test is the only thing preventing the two implementations
from drifting apart again.

### A working session that cannot infer

ONNX Runtime created a `CUDAExecutionProvider` session successfully, passed a
provider check, and then failed **every** inference with `NOT_IMPLEMENTED`,
because cuDNN is resolved lazily at first kernel launch. The service would have
started, reported itself healthy, and returned 503 for all traffic.

**Session creation does not prove a backend works.** Every candidate now runs a
warmup inference before being accepted, which turns a permanent runtime failure
into a load-time fallback — the thing that makes "graceful degradation" real
rather than aspirational.

### Every metric labelled `unmatched`

The metrics middleware resolved the route template *before* `call_next`, but
Starlette only populates `scope["route"]` during routing. Every request was
labelled `endpoint="unmatched"`, collapsing `http_requests_total` into a single
series. Metrics existed and looked healthy, which is exactly what made it
invisible — and it made every per-endpoint dashboard panel and latency alert
useless.

### A cache that could fail a request

`CacheService.set` caught `RedisError` and `TypeError`, but `json.dumps` raises
`ValueError` on a circular reference. An unserialisable value would have
propagated into the request path, breaking the fail-open guarantee the cache is
built around. Found by a test that deliberately supplied a cyclic structure.

---

## 5. Testing

276 tests, 91.2% statement coverage of `api/`, running in under eight seconds.

**Redis is faked, not mocked.** `fakeredis` implements real Redis semantics
including Lua evaluation, so the rate limiter's sliding-window script executes
as written. The atomicity test fires four times the quota concurrently and
asserts exactly the limit is admitted; it fails against a non-atomic
implementation.

That decision caught the most instructive failure in the project. `fakeredis`
without the `lua` extra silently lacks an interpreter: `EVALSHA` fails, the
limiter correctly fails open, and **every rate-limiting test would have passed
vacuously**. A green suite that proves nothing is the most dangerous failure
mode in testing, and it is only caught by checking that tests fail when they
should.

**Tests do not depend on the 87 MB model.** A synthetic ONNX graph with the same
interface is generated per session, so CI needs no artefacts.

**Security behaviours are asserted**: `alg=none`, algorithm confusion,
missing-expiry tokens, API key prefix probing, constant-time lookup, a
sub-100-byte decompression bomb, Content-Type confusion, SSRF via non-HTTP batch
URLs, and that no error message leaks paths or stack frames.

Three of my own tests were wrong before they were right: one assumed TensorRT
was unavailable (this machine has it, so the test measured the host); one
asserted a batching property the synthetic model cannot exhibit; one compared
megabytes for a kilobyte-sized fixture. Environment-dependent tests are worse
than no test.

---

## 6. Scalability

**Stateless API.** No per-request state outside Redis and Postgres, so replicas
scale horizontally with no coordination. The production overlay runs three with
rolling updates and automatic rollback.

**Batching is where throughput comes from.** 760 img/s at batch 1 against
6,118 at batch 32 — a single image cannot saturate the GPU. The async batch
endpoint exists to exploit this.

**The forward pass runs in a thread pool.** ONNX Runtime's `Run()` blocks;
executing it on the event loop would stall every in-flight request for its
duration. Asserted by a test that checks a lightweight coroutine still advances
during inference.

**Known bottlenecks, in the order they would bite:**

1. **Model memory per worker.** Each Celery worker process loads its own copy.
   Concurrency is capped at 2 for this reason; raising it multiplies memory.
2. **Postgres write volume.** One row per inference. At sustained high traffic
   this needs partitioning by time and a retention policy — neither is
   implemented.
3. **Redis as a single point of contention.** Cache, rate limits, and the Celery
   broker share one instance. They should be separated before scaling far.
4. **No autoscaling.** Replica counts are static.

**Cache effectiveness** is measured: a hit is ~45× faster (9.0 ms → 0.2 ms).
Value depends entirely on repeat-submission rate, which the `inference_logs`
image digest column is there to measure.

---

## 7. Model validation

Three components in `models/validation/`, all operating on the `inference_logs`
audit trail.

### Drift detection

Two statistics, chosen for different variable types:

**PSI** for categorical features (predicted class, image format). The industry
convention, with established interpretation bands (<0.1 none, 0.1-0.25
moderate, >0.25 significant) and insensitive to sample size in the way a
hypothesis test is not. The epsilon guard matters: a category present in one
window and absent in the other otherwise yields infinite PSI, and that is
exactly the case worth detecting.

**Kolmogorov-Smirnov** for continuous features (dimensions, latency).
Distribution-free and needs no binning. **Severity is driven by the KS
statistic, never the p-value** — at 50,000 samples per window a hypothesis test
rejects the null for differences far too small to matter, so a p-value-driven
alert would report permanent drift. Asserted by a test that feeds two
distributions differing by 0.3 in 50,000 samples and requires severity `none`.

Prediction drift is the more practical signal, because it needs no labels.
Ground truth for production traffic arrives late or never, so accuracy cannot be
monitored directly; a sustained change in *what the model predicts* is the
earliest available warning.

### A/B testing

**Assignment is deterministic per user**, by hashing `experiment + user_id`. A
user stays on the arm they were assigned. Random per-request assignment would
give the same caller different versions for identical inputs, and would
correlate observations within a user across arms — violating the independence a
significance test assumes. The experiment name is in the hash so a user unlucky
in one test is not systematically in every challenger arm; verified at ~50%
cross-experiment agreement.

Comparisons report **effect size and confidence intervals alongside p-values**,
and a change must be both significant *and* materially large before it blocks a
rollout. Verified: a 0.5% latency difference at n=50,000 is reported significant
(p ≈ 0) and immaterial, with the recommendation "no material difference". A
significance-only gate would block every release.

Welch's t-test rather than Student's, because arms have no reason to share a
variance — a slower backend is usually also more variable.

### Regression gates

Per-metric, directional tolerances against a **committed** baseline. Accuracy
permits 2% relative degradation with an absolute floor of 80%; latency permits
20%, deliberately loose because CI runners are noisy and a gate that produces
false failures trains people to ignore it.

The absolute floor exists because a relative tolerance alone permits gradual
erosion — each release individually within tolerance while quality slides across
many. The baseline is committed and updated only by an explicit, reviewed step,
for the same reason: a baseline recomputed from recent runs drifts upward and
quality erodes without any single comparison ever failing.

A **missing** metric fails the check. Silently passing a comparison that could
not be made would let an unmeasured regression through, which is the opposite of
what a gate is for.

---

## 8. What is missing, and why

Stated plainly rather than omitted.

| Gap | Reason |
| --- | --- |
| Detector evaluation | Published COCO mAP is quoted; no independent evaluation run |
| Calibration and per-class metrics | Only aggregate accuracy measured |
| Similarity precision@5 across all classes | Measured on 5 of 200; per-class variation is demonstrably large |
| GPU in containers | Compose runs ONNX Runtime on CPU (~170 ms); GPU passthrough needs device reservations and `onnxruntime-gpu` |
| Postgres partitioning and retention | Needed before sustained production write volume |
| Live A/B traffic splitting in the API | The framework and analysis exist; the serving path does not yet route by variant |
| Automated drift alerting | `scripts/analyse_drift.py` runs on demand; no scheduled job or alert wiring |

The last two are the natural next step: the assignment logic, the audit schema,
and the comparison statistics all exist, and wiring them into the request path
plus a scheduled job is integration work rather than new design.
