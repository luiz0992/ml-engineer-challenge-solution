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

### What was not built

A third model (similarity search over classifier embeddings) was scoped but not
implemented. The brief asks for three; two are delivered. Stating that plainly
is more useful than a half-working third.

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

## 7. What is missing, and why

Stated plainly rather than omitted.

| Gap | Reason |
| --- | --- |
| Third model (similarity search) | Scoped, not reached |
| Drift detection, A/B framework | `models/validation/` is an empty package; the audit schema was designed for it |
| Performance regression gates | No recorded baseline to compare against |
| Detector evaluation | Published COCO mAP is quoted; no independent evaluation run |
| Calibration and per-class metrics | Only aggregate accuracy measured |
| GPU in containers | Compose runs ONNX Runtime on CPU (~170 ms); GPU passthrough needs device reservations and `onnxruntime-gpu` |
| Postgres partitioning and retention | Needed before sustained production write volume |

The drift and A/B work is the most valuable of these, and the schema and metrics
to support it already exist — `inference_logs` records the input fingerprint,
predicted class, and model version needed to compare distributions across
versions over time.
