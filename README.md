# Multi-Model Computer Vision API

A production-oriented MLOps system that serves three computer-vision models —
image classification, object detection, and image similarity search — behind a
single authenticated, rate-limited, observable HTTP API.

Three models served: image classification, object detection, and image
similarity search. See
[What is still missing](docs/technical-writeup.md#12-what-is-still-missing) for an
explicit list of remaining gaps.

---

## Contents

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Notes on the provided starter scripts](#notes-on-the-provided-starter-scripts)
- [Results](#results)
- [Testing](#testing)
- [Scripts](#scripts)
- [Known limitations](#known-limitations)

---

## Architecture

```
                    ┌──────────────┐
   client ─────────▶│ api-gateway  │  nginx: per-IP rate limits, upload cap,
                    │   (nginx)    │  security headers, /metrics denied
                    └──────┬───────┘
                           │ frontend network
                    ┌──────▼───────┐
                    │    ml-api    │  FastAPI: auth, validation, per-tier
                    │   (uvicorn)  │  rate limits, ONNX inference
                    └──┬────────┬──┘
           backend ────┤        ├──── backend
              ┌────────▼──┐  ┌──▼────────┐
              │   redis   │  │ postgres  │
              │ cache +   │  │ inference │
              │  broker   │  │   logs    │
              └─────┬─────┘  └───────────┘
                    │ broker
              ┌─────▼─────┐        ┌────────────┐     ┌──────────┐
              │  worker   │        │ prometheus │────▶│ grafana  │
              │ (celery)  │        │  + alerts  │     │dashboards│
              └───────────┘        └────────────┘     └──────────┘
```

Networks are segmented: Postgres and Redis sit on `backend` only, so they are
unreachable from anything exposed to the outside world. The API is not
published to the host — traffic arrives through the gateway, which is also what
enforces the `/metrics` deny rule.

### Running the stack

```bash
cp .env.example .env          # set JWT_SECRET_KEY and POSTGRES_PASSWORD
uv run python scripts/prepare_artifacts.py   # publish model artifacts
docker compose up -d --build

curl localhost:8080/api/v1/health
```

Twelve services. The ones you interact with: `api-gateway` (`:8080`), `ml-api`,
`worker`, `redis`, `postgres`, `prometheus` (`:9090`), `grafana` (`:3000`,
admin/admin), `alertmanager`. The rest run unattended: `migrate` applies
migrations once at startup, `maintenance` and `drift` are scheduled jobs, and
`pushgateway` collects metrics from those jobs.

Production overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

The overlay adds replicas, read-only root filesystems, dropped capabilities,
and removes the published Prometheus and Grafana ports. It has **no default
secrets** — `JWT_SECRET_KEY`, `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, and
`GRAFANA_PASSWORD` must be supplied or Compose refuses to start, so a
production deploy cannot silently fall back to a development credential.

### Inference audit trail

Every inference is recorded to Postgres: caller, model name, version, backend,
latency, cache status, a SHA-256 of the input, and the top prediction. This
answers questions metrics cannot, because Prometheus aggregates and discards
individual events — which model version produced a disputed prediction, what
the input distribution looked like last Tuesday, which user caused a latency
spike.

Two properties govern the design. **A database problem never fails a request**:
the inference succeeded, so the user gets their answer whether or not we
recorded it. **Recording adds no latency**: writes are queued and flushed by a
background task in batches, because a synchronous insert would add a network
round trip to a request whose entire budget is a few milliseconds. The queue is
bounded, so a database outage degrades the audit trail rather than growing
memory until the process is OOM-killed.

No image bytes are stored, only a digest. Images belong in an object store with
its own retention and access-control policy, not in an operational database.

Schema changes go through Alembic. Migrations run as a one-shot `migrate`
service that must complete before the API starts, rather than from the API
entrypoint — with several replicas, an entrypoint migration means N processes
racing to apply the same DDL.

```bash
docker compose run --rm migrate alembic upgrade head   # apply
docker compose run --rm migrate alembic downgrade -1   # roll back one
```

### Image design

One Dockerfile, two targets (`api` and `worker`) sharing a `runtime` stage, so
dependency layers are built once. The serving image is 701 MB and contains
**no torch, torchvision, or CUDA** — preprocessing is reimplemented in NumPy and
Pillow precisely so the training stack can be left out. Both run as a non-root
user (uid 1001), and model artifacts are mounted read-only rather than baked in,
so a new model version needs no rebuild.

### Kubernetes

```bash
kubectl apply -f deploy/kubernetes/
```

Includes a HorizontalPodAutoscaler targeting `http_requests_in_progress` per
pod rather than CPU — inference saturates the GPU or the thread pool well
before CPU looks busy, so a CPU-targeted autoscaler scales after latency has
already degraded. Scheduled work runs as CronJobs with retry policies and run
history, which the sleeping containers Compose requires cannot provide.

### GPU and scheduled maintenance

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

The GPU overlay adds device reservations and installs `onnxruntime-gpu` plus
TensorRT, taking single-image classification from 15.9 ms (ONNX Runtime CPU in
the container) to 1.05 ms (TensorRT FP16). It is a separate
overlay because a Compose file that demands a GPU fails outright on a machine
without one.

The `maintenance` service creates upcoming monthly partitions and drops expired
ones daily. `inference_logs` is partitioned by month: dropping a partition is
instant where deleting a month of rows is a long-running statement that bloats
the table. Retention defaults to six months.

### Troubleshooting

**`Temporary failure resolving deb.debian.org` during build.** BuildKit cannot
reach DNS on some Docker Desktop configurations, while `docker run` containers
can. Build with the host network:

```bash
docker build --network=host --target api -t mlchal-api:latest .
docker build --network=host --target worker -t mlchal-worker:latest .
docker compose up -d --no-build
```

**GPU inference in containers.** The Compose stack runs ONNX Runtime on CPU:
measured in the running container, 15.9 ms per classification and 73 ms per
detection. GPU passthrough needs `deploy.resources.reservations.devices`
and the `onnxruntime-gpu` package in the image; see `benchmarks/ANALYSIS.md` for
the measured difference.

## Results

### Image classification — Tiny-ImageNet (200 classes)

Fine-tuned `vit_small_patch16_224.augreg_in21k_ft_in1k` for 5 epochs on a
single RTX 5000 Ada.

| Metric | Value |
| --- | --- |
| Top-1 accuracy | **85.88%** |
| Top-5 accuracy | **96.54%** |
| Validation loss | 0.5965 |
| Training time | 7 min 4 s (5 epochs) |
| Throughput | ~9.7 optimizer steps/s at batch 128 |

Measured over the full 10,000-image validation split. Accuracy improved
monotonically across all five epochs with no divergence.

| Epoch | Train loss | Val loss | Top-1 | Top-5 |
| --- | --- | --- | --- | --- |
| 0 | 3.222 | 0.738 | 82.37% | 95.56% |
| 1 | 2.437 | 0.664 | 84.17% | 96.18% |
| 2 | 2.315 | 0.619 | 85.28% | 96.50% |
| 3 | 2.213 | 0.603 | 85.80% | 96.54% |
| 4 | 2.109 | 0.596 | **85.88%** | **96.54%** |

Reproduce with:

```bash
python scripts/setup/download_datasets.py --dataset tiny_imagenet
uv run python -m models.training.train
```

### Object detection — COCO (80 classes)

RT-DETR R18, Apache-2.0, used pretrained.

| Metric | Value |
| --- | ---: |
| **mAP@[.5:.95]** | **0.500** (measured on COCO val2017) |
| mAP@0.5 | 0.667 |
| mAP small / large | 0.347 / 0.627 |
| Latency (TensorRT FP16) | **2.14 ms** at batch 1, 1.04 ms/image at batch 8 |

Evaluated with `pycocotools`, not quoted from the publication. The
small-object gap of 0.280 is the model's real
weakness.

Licensing drove the model choice: Ultralytics YOLOv8/v11 are AGPL-3.0, which
obliges anyone offering the service over a network to publish their source.

### Image similarity search

The fine-tuned classifier backbone with its head removed, indexed with FAISS
over 20,000 training images.

| Metric | Value |
| --- | --- |
| Precision@1 | 81.5% |
| **Precision@5** | **79.7%** |
| Query latency | ~5 ms |
| Embedding | 384-d, L2-normalised, cosine similarity |

Measured over 1,000 validation queries drawn at random from the validation
split, which happened to cover 198 of the 200 classes. Per-class
variation is large (19.2pp sd): context-defined categories like `pole` retrieve
at 10%, distinctive ones near-perfectly.

Retrieves **semantically** similar images rather than near-duplicates —
classification features collapse intra-class variation by construction. For
near-duplicate detection a perceptual hash would be the right tool.

### Measured quality

`scripts/evaluate_models.py` reports per-class accuracy and calibration.

| | |
| --- | --- |
| Per-class accuracy | mean 85.8%, sd 8.6pp, range 56–100%, **none below 50%** |
| Calibration (ECE) | 0.0852, **underconfident by 0.085** |

The classifier reports 77.3% mean confidence while being right 85.8% of the
time. Thresholding at 0.9 discards many predictions that are correct 96% of the
time; **0.75 is the right threshold for ~95% precision**.

### Model validation

`models/validation/` provides drift detection, A/B testing, and regression
gates, all operating on the inference audit trail. Drift results are pushed to
Prometheus so they alert through the same routing as everything else.

```bash
uv run python scripts/analyse_drift.py --baseline-days 7 --current-days 1
uv run python scripts/check_regression.py
uv run python scripts/evaluate_models.py
```

**A/B testing is wired into the serving path.** An experiment file at
`models/artifacts/experiments.json` splits traffic across model versions; the
assigned variant is recorded in the audit trail so the arms can be compared.
An explicit `?model_version=` always overrides an experiment, and a malformed
experiment degrades to the active version rather than failing requests.

### Model versioning

A version is resolved from the artefact layout, so shipping a second one is a
deployment step rather than a code change:

```
models/artifacts/onnx/
├── classifier_fp32.onnx          # v1, flat layout
├── labels.json                   # (at the artifacts root)
└── v2/
    ├── classifier_fp32.onnx      # v2 weights
    └── labels.json               # and v2's own class list
```

**A non-default version must be self-describing.** Its directory carries its
own `labels.json`, so a `v2` built on a re-ordered or extended class list
cannot be named from `v1`'s mapping — versioning the weights without
versioning what they mean mislabels every prediction while the response, the
audit row and the A/B analysis all agree.

Every version present on disk is loaded at startup and registered, but **not**
made active — a new version becomes reachable by pinning it or by an
experiment, never by the mere act of shipping it. Callers pin with
`?model_version=v2`, and the version that served is returned in `provenance`
and written to the audit trail.

**A pinned version that is not on disk fails rather than falling back.** That
is the whole point of resolving through the layout: serving `v1`'s weights
under `v2`'s name would make the response, the provenance block, the audit row,
and the A/B analysis all agree on a version that never ran. The rollout
procedure is in the [operations runbook](docs/operations.md#shipping-a-new-model-version).

Drift uses PSI for categorical features and Kolmogorov-Smirnov for continuous
ones, with **severity driven by effect size rather than p-value** — at
production volumes a hypothesis test reports permanent drift. A/B comparisons
likewise require a difference to be both statistically significant *and*
materially large before it blocks a rollout.

### Inference performance

ViT-Small/16 at 224x224, batch 32, RTX 5000 Ada. Full matrix in
[`benchmarks/README.md`](benchmarks/README.md); analysis and deployment
recommendation in [`benchmarks/ANALYSIS.md`](benchmarks/ANALYSIS.md).

<!-- BEGIN:headline-latency -->
<!-- Generated by scripts/sync_docs.py; edits here are overwritten. -->

| Backend | Latency | Throughput | Speedup |
| --- | ---: | ---: | ---: |
| PyTorch eager FP32 | 16.73 ms | 1,913 img/s | 1.00x |
| PyTorch eager bf16 | 5.19 ms | 6,162 img/s | 3.22x |
| **TensorRT FP16** | **3.51 ms** | **9,105 img/s** | **4.76x** |
| ONNX Runtime CUDA FP32 | 18.94 ms | 1,689 img/s | 0.88x |
| ONNX Runtime CPU INT8 | 460.80 ms | 69 img/s | 0.04x |

<!-- END:headline-latency -->

All three models have TensorRT engines: classifier 9,105 img/s and embedder
9,192 img/s at batch 32, detector 965 img/s at batch 8 (it runs at 8x the
input pixels, so large batches exhaust GPU memory before saturating compute).

Single-image latency is 1.05 ms under TensorRT, three orders of magnitude
inside the sub-second requirement.

Accuracy is reported separately from latency because the two are measured on
different splits. The headline **top-1 is 85.78%** on the full 10,000-image
validation split (`benchmarks/evaluation.json`). The INT8 calibration study in
[`benchmarks/ANALYSIS.md`](benchmarks/ANALYSIS.md) uses a fixed 2,000-image
subset (87.25% FP32 -> 81.85% INT8); the subset figures are comparable to each
other but are not the model's accuracy.

**INT8 is built for all three models and recommended for none.** Each was
quantized with the same pipeline and measured with the metric it is judged on:

| Model | Metric | FP32 | INT8 | Change |
| --- | --- | ---: | ---: | ---: |
| Classifier | Top-1 | 87.25% | 81.85% | -5.40pp |
| Detector | COCO mAP@[.5:.95] | 0.4999 | 0.0632 | -87% rel. |
| Embedder | recall@5 vs FP32 index | 1.000 | 0.505 | -50% rel. |

The detector's small-object AP falls to *exactly zero* — box regression has
none of the margin a classifier's argmax enjoys — and it keeps emitting ~246k
detections, so it fails confidently rather than silently. Reaching even -5.4pp
on the classifier required replacing ONNX Runtime's default min-max
calibration, which cost 17.7 points, with percentile calibration: vision
transformers produce rare LayerNorm and GELU outliers that destroy a min-max
range. Full study in [`benchmarks/ANALYSIS.md`](benchmarks/ANALYSIS.md) §2.

### Memory footprint

Measured by `scripts/profile_memory.py` on a CUDA host; full report in
[`benchmarks/memory.md`](benchmarks/memory.md).

| Component | Resident |
| --- | ---: |
| ONNX Runtime + CUDA context (one-time) | 839.5 MB |
| Classifier | 79.0 MB |
| Detector | 226.5 MB |
| Embedder | 87.6 MB |
| **Total, all three models loaded** | **1303 MB** |

Two things worth stating. **The runtime dominates**: CUDA context and cuDNN
kernels cost more than twice the combined model weights, and that cost is paid
once regardless of how many models are resident. Sizing a container from the
260 MB of model files alone would under-provision it fivefold. This is also why the default
serving image is CPU-only — the CUDA stack is opt-in via
`docker-compose.gpu.yml`.

**Per-request memory is flat in upload size.** Peak Python allocation is
1.79 MB (p95 1.80 MB) and does not move between a 224x224 and a 2048x2048
upload, because the image is resized before anything else touches it. A peak
that tracked the upload would make a large image a memory-exhaustion vector
rather than a validation concern; two performance tests assert it stays flat.

**Why training loss stays near 2.1 while validation loss is 0.60.** These are
not the same quantity. Training loss is measured against MixUp/CutMix-mixed
soft targets with label smoothing, which carry irreducible entropy — a
perfectly calibrated model cannot drive it to zero. Validation loss is measured
against clean, unmixed labels. The gap is expected and is evidence the
regularisers are active, not of a bug.

**Why the backbone transfers so well.** Tiny-ImageNet's 200 classes are a
subset of ImageNet-1k, so this backbone has already seen every target category
during pretraining. This is a genuine advantage of the model choice rather than
a property of the method, and is stated explicitly here because it makes the
headline number less impressive than it first appears.


## Quick start

Requires Docker with the NVIDIA container runtime (for GPU inference) and
[uv](https://docs.astral.sh/uv/) for local development.

```bash
cp .env.example .env          # then edit; JWT_SECRET_KEY must be set
uv sync --extra dev           # serving + test dependencies
uv sync --extra train         # adds torch/CUDA, only needed for training
```

## API

Eleven paths under `/api/v1`. The health and metrics endpoints are also
mounted unprefixed, so orchestrator probes need not know the API version.

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/api/v1/auth/token` | API key | Exchange an API key for a bearer token |
| POST | `/api/v1/classify` | Bearer | Classify a single image |
| POST | `/api/v1/detect` | Bearer | Object detection (80 COCO classes) |
| POST | `/api/v1/similar` | Bearer | Find visually similar images |
| POST | `/api/v1/batch` | Bearer | Submit an async batch job (202 + job ID) |
| GET | `/api/v1/batch/{job_id}` | Bearer | Poll job progress and results |
| GET | `/api/v1/models` | none | Registered models and metadata |
| GET | `/api/v1/health` | none | Aggregate health with per-component detail |
| GET | `/api/v1/metrics` | none | Prometheus metrics |
| GET | `/health/live`, `/health/ready` | none | Liveness and readiness probes |

Interactive documentation is generated at `/docs`; the OpenAPI schema is at
`/openapi.json`.

### Example

```bash
TOKEN=$(curl -s localhost:8000/api/v1/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"dev-key-pro"}' | jq -r .access_token)

curl -s localhost:8000/api/v1/classify?top_k=3 \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@image.jpeg | jq
```

```json
{
  "predictions": [
    {"label": "goldfish", "class_id": 0, "wnid": "n01443537", "probability": 0.959}
  ],
  "inference_time_ms": 8.8,
  "correlation_id": "7a3bade2604f4776a0bfa8d136030e61",
  "provenance": {"model_name": "tiny-imagenet-classifier", "model_version": "v1", "backend": "onnx"},
  "cached": false
}
```

### Rate limits

Enforced per tier with a sliding window in Redis, reported in
`X-RateLimit-Limit` and `X-RateLimit-Remaining` on every response.

| Tier | Requests / minute |
| --- | --- |
| free | 10 |
| pro | 120 |
| enterprise | 1200 |

### Errors

Every error returns the same envelope, with a stable `code` to branch on and
the `correlation_id` to quote when reporting a problem.

```json
{"error": {"code": "unsupported_format", "message": "Format TIFF is not supported.",
           "details": {"supported_formats": ["BMP", "JPEG", "PNG", "WEBP"]},
           "correlation_id": "..."}}
```

## Testing

```bash
scripts/test                 # unit + integration, coverage gate at 90%
scripts/test --performance   # performance and memory suite
scripts/test --fast          # skip coverage, for a tight edit loop
scripts/lint --fix           # ruff + mypy
```

**431 tests, 91.6% combined statement and branch coverage** of `api/`. CI gates
at 90%.

| Suite | Count | Scope |
| --- | ---: | --- |
| Unit | 344 | Validation, preprocessing, auth, rate limiting, cache, model loading |
| Integration | 75 | Full request path with a fake Redis and a synthetic model, plus database operations against a real Postgres |
| Performance | 12 | Latency, memory profiling, concurrency, batching |

Most of the suite runs against fakes — `fakeredis` with real Lua semantics, a
synthetic ONNX graph — because they are faithful and fast. The database is the
exception: `tests/integration/test_database.py` runs against a real Postgres
(supplied by CI, or started with testcontainers locally, or skipped if neither
is available). Timezone handling, partial indexes, and a write that fails
*slowly* cannot be observed against a stand-in, and the third of those turned
out to matter — see [the audit-shutdown
bug](docs/technical-writeup.md#a-batch-lost-between-the-queue-and-the-database).

### Load and stress testing

```bash
docker compose up -d
uv run locust -f tests/performance/locustfile.py --host http://localhost:8080 \
    --headless --users 50 --spawn-rate 5 --run-time 2m
```

Three profiles: `ClassificationUser` (the dominant pattern, with think time),
`BatchUser` (submit-and-poll), and `FreeTierUser` (saturates the rate limiter
to confirm rejection stays cheap and carries `Retry-After`). The run exits
non-zero if the failure ratio exceeds 1% or p95 exceeds 2 s, so CI can gate on
it.

A measured run against the full Compose stack is committed:
[`benchmarks/load-test.md`](benchmarks/load-test.md) — 1,361 requests through
the nginx gateway at 20 concurrent users, **zero failures**, `/classify` p50
23 ms and p95 67 ms, `/detect` p50 96 ms. Every percentile is an order of
magnitude inside the sub-second requirement, and the cache is worth about 8x
(3 ms against 23 ms).

### Memory profiling

```bash
uv run python scripts/profile_memory.py
```

Writes `benchmarks/memory.json` and `benchmarks/memory.md`: resident cost per
model, peak allocation per request, where allocations are retained, and how
peak scales with upload size. See [results](#memory-footprint).

Performance tests are excluded by default: they are slower and their thresholds
depend on the host, so they belong in a deliberate run rather than in the loop a
developer repeats every few minutes.

Two testing decisions worth stating:

**Redis is faked, not mocked.** `fakeredis` implements real Redis semantics
including Lua evaluation, so the rate limiter's sliding-window script executes
as written. A `MagicMock` would assert only that we call the methods we call,
which proves nothing about whether the algorithm is correct. The atomicity test
fires four times the quota concurrently and asserts exactly the limit is
admitted — that test fails against a non-atomic implementation.

**Tests do not depend on the 87 MB trained model.** A synthetic ONNX graph with
the same interface is generated at session scope, so the suite runs in CI with
no artifacts. Tests that genuinely need the trained model skip with a reason
when it is absent.

The most important single test is
`tests/unit/test_image_processing.py::TestTorchvisionEquivalence`, which asserts
the serving preprocessing matches the training transform to 7.2e-07 across seven
input shapes. It exists because two real bugs were found that way, both silent:
`v2.Resize` defaults to BILINEAR rather than BICUBIC, and `CenterCrop` rounds
rather than floor-divides. Since the serving path deliberately reimplements
preprocessing in NumPy so the production image needs no torch, that test is the
only thing preventing the two implementations from drifting apart again.

## Project layout

```
api/          FastAPI application (routers, services, middleware, schemas)
models/       Dataset pipelines, training, optimisation, model validation
worker/       Celery application and batch inference tasks
tests/        Unit, integration, and performance suites
monitoring/   Prometheus scrape config, alert rules, Grafana dashboards
deploy/       Kubernetes manifests
docker/       nginx gateway configuration (the Dockerfile is at the repo root)
scripts/      Dataset download and operational utilities
benchmarks/   Generated performance comparison reports
docs/         Model cards and the technical write-up
```

## Scripts

Every measurement quoted in this repository is produced by one of these, and
each writes its output under `benchmarks/` so the claim can be re-derived
rather than trusted.

| Script | Produces |
| --- | --- |
| `scripts/setup/download_datasets.py` | Tiny-ImageNet and a COCO val2017 subset |
| `scripts/prepare_artifacts.py` | ONNX exports, INT8 quantization, label metadata |
| `scripts/build_similarity_index.py` | The FAISS index backing `/api/v1/similar` |
| `scripts/evaluate_models.py` | `benchmarks/evaluation.json` — per-class accuracy, calibration, retrieval |
| `scripts/evaluate_detector.py` | `benchmarks/detection_eval*.json` — COCO mAP, off-distribution probe |
| `scripts/verify_exports.py` | `benchmarks/export_fidelity.json` — ONNX-vs-PyTorch agreement |
| `scripts/profile_memory.py` | `benchmarks/memory.{json,md}` — resident cost, peak allocation |
| `scripts/check_regression.py` | Gates a run against the committed baseline; exits non-zero on regression |
| `scripts/analyse_drift.py` | PSI and KS over the audit trail; pushes gauges to Pushgateway |
| `scripts/maintain_partitions.py` | Creates and drops audit partitions |
| `scripts/sync_docs.py` | Regenerates the benchmark tables embedded in the docs; `--check` in CI |

Plus `models/optimisation/run_benchmarks.py` for the latency matrix and
`tests/performance/locustfile.py` for load.

Operational usage — deploying, scaling, shipping a model version, responding to
an alert — is in the [operations runbook](docs/operations.md).

## Notes on the provided starter scripts

The challenge repository ships helper scripts under `scripts/`. Two contain
defects that would corrupt results if used as-is, so this project supplies
corrected implementations under `models/` and documents the originals here.

### 1. `download_datasets.py` cannot run

```python
parser.add_argument("--dataset", choices=list(DatasetDownloader({}).datasets.keys()) + ["all"], ...)
```

`DatasetDownloader({})` forwards a `dict` to `Path()`, raising `TypeError` while
argparse is still being constructed. The script therefore fails on every
invocation, including the exact commands given in the challenge README. The
dataset catalogue is class state that does not depend on `data_dir`, so the fix
is to lift it to a module-level constant rather than instantiate a throwaway
object.

### 2. `tiny_imagenet_dataloader.py` silently mislabels the validation set

```python
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_val)
```

Tiny-ImageNet's validation split is **not** in `ImageFolder` layout. It is a
flat `val/images/` directory plus a `val_annotations.txt` file mapping each
filename to its class. `ImageFolder` consequently discovers exactly one class
(`images`) and assigns label `0` to all 10,000 validation images.

This fails *silently* — it raises no error, and reported validation accuracy
becomes meaningless. `models/data/` restructures the validation split into proper
per-class directories before loading.

## Known limitations

The full list, with reasoning, is in
[technical write-up §12](docs/technical-writeup.md#12-what-is-still-missing).
The ones worth knowing before you read the results:

| Limitation | Detail |
| --- | --- |
| INT8 is built for all three models and deployed for none | Quantization is applied and measured per model. It costs the classifier 5.4pp of top-1, the detector 87% of its mAP (small-object AP falls to exactly zero), and the embedder half its top-5 retrieval. Serving runs FP32 ONNX with TensorRT FP16 engines. See [`benchmarks/ANALYSIS.md`](benchmarks/ANALYSIS.md) §2. |
| The embedder is not a third network | It is the fine-tuned classifier backbone with the head removed. Cheap to serve and honest about it, but a purpose-trained metric-learning model would retrieve better. |
| Detector accuracy off-distribution | Unmeasured — mAP needs annotations no other dataset here provides. Its failure *mode* is characterised: it abstains rather than hallucinating. |
| Alert delivery endpoints | Routing, grouping, and inhibition are configured and validated with `amtool`. The webhook and PagerDuty keys belong in a secret manager, not a repository. |
| Kubernetes manifests not applied to a live cluster | Validated against real 1.30 API schemas with `kubeconform --strict` (10/10 resources), which is a smaller claim than "deployed and working". |
| Only one model version is built | Versioning is resolved from the artefact layout and a second version needs no code change, but the pipeline produces `v1` only. There is no second set of weights in this repository to demonstrate a live rollout against. |

## Documentation

Start with the [documentation index](docs/README.md), which carries every
measured result in one table plus a catalogue of the defects found during
development and how each was caught.

| Document | Contents |
| --- | --- |
| [Technical write-up](docs/technical-writeup.md) | Model selection, optimisation results, architecture decisions, scalability, and an explicit list of gaps |
| [Operations runbook](docs/operations.md) | Deploying, scaling, shipping a model version, running an experiment, alert responses |
| [Model card: classifier](docs/model-card-classifier.md) | Metrics, training procedure, limitations, ethical considerations |
| [Model card: detector](docs/model-card-detector.md) | RT-DETR provenance, licensing rationale, limitations |
| [Model card: embedder](docs/model-card-embedder.md) | Retrieval quality, index construction, a published correction |
| [Benchmark results](benchmarks/README.md) | Generated latency matrix across backends |
| [Benchmark analysis](benchmarks/ANALYSIS.md) | Interpretation and deployment recommendation |
| [Load test](benchmarks/load-test.md) | End-to-end latency through the full stack under concurrency |
| [Memory profile](benchmarks/memory.md) | Resident cost per model, peak allocation per request |
| [OpenAPI spec](docs/openapi.json) | Exported schema; also served live at `/openapi.json` |
| [Documentation index](docs/README.md) | All results in one place, and what is not done |

Eleven silent bugs found during development are written up in
[section 4](docs/technical-writeup.md#4-eleven-bugs-worth-reporting):

- two preprocessing defects that shifted every activation and crop;
- a backend that created a working session and then failed every inference;
- a metrics mislabelling that made every dashboard panel useless;
- a container that could not see its own CPU limit and ran **8x slow**;
- an audit writer that lost the batch it was committing on every shutdown;
- version pinning that served `v1`'s weights under `v2`'s name;
- a drift job that could not import the library its only test needs;
- a provider check that skipped itself in the one case it existed for,
  reporting `backend=tensorrt` while running on CPU;
- versioned weights served with another version's labels;
- a cache that could fail a request it was meant to accelerate.

Four of the last five were found by running the system rather than testing it
— `docker compose up` and a stopwatch, a real Postgres instead of a fake, a
populated database that let the drift job reach its own test, and an artefact
layout that made the versioning claim checkable.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs seven jobs:

| Job | What it guards |
| --- | --- |
| `lint` | ruff check, ruff format, mypy across `api models worker scripts tests` |
| `test` | Real Postgres and Redis services, 90% coverage gate, migration up/down/up, and a check that no test skipped unexpectedly |
| `preprocessing-equivalence` | Installs torch so the serving-vs-training equivalence test actually runs, and fails if it skips |
| `model-quality` | Validates the committed baseline, then gates on metric regressions |
| `docs-contract` | Fails if `docs/openapi.json` or the generated benchmark tables have drifted |
| `security` | `pip-audit` (advisory) and a blocking secret scan |
| `docker` | Builds both images and asserts non-root, no torch, under 1.5 GB, plus kubeconform / promtool / amtool / Compose validation |

## Licence

MIT. Third-party model weights retain their upstream licences; see the model
cards in [`docs/`](docs/).
