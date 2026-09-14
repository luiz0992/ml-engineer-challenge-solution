# Multi-Model Computer Vision API

A production-oriented MLOps system that serves three computer-vision models —
image classification, object detection, and image similarity search — behind a
single authenticated, rate-limited, observable HTTP API.

Three models served: image classification, object detection, and image
similarity search. See
[What is missing](docs/technical-writeup.md#8-what-is-missing-and-why) for an
explicit list of remaining gaps.

---

## Contents

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Notes on the provided starter scripts](#notes-on-the-provided-starter-scripts)
- [Design decisions](#design-decisions)

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

Services: `api-gateway`, `ml-api`, `worker`, `redis`, `postgres`, `prometheus`
(`:9090`), `grafana` (`:3000`, admin/admin).

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
dependency layers are built once. The serving image is 700 MB and contains
**no torch, torchvision, or CUDA** — preprocessing is reimplemented in NumPy and
Pillow precisely so the training stack can be left out. Both run as a non-root
user (uid 1001), and model artifacts are mounted read-only rather than baked in,
so a new model version needs no rebuild.

### Troubleshooting

**`Temporary failure resolving deb.debian.org` during build.** BuildKit cannot
reach DNS on some Docker Desktop configurations, while `docker run` containers
can. Build with the host network:

```bash
docker build --network=host --target api -t mlchal-api:latest .
docker build --network=host --target worker -t mlchal-worker:latest .
docker compose up -d --no-build
```

**GPU inference in containers.** The Compose stack runs ONNX Runtime on CPU
(~170 ms per image). GPU passthrough needs `deploy.resources.reservations.devices`
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
| Peak GPU memory | 5.4 GiB |

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

RT-DETR R18, Apache-2.0, used pretrained. **15 ms** per image. Verified on COCO
`000000039769`: two cats, two remotes, and a sofa, all at 0.74–0.95 confidence.

Licensing drove the model choice: Ultralytics YOLOv8/v11 are AGPL-3.0, which
obliges anyone offering the service over a network to publish their source.

### Image similarity search

The fine-tuned classifier backbone with its head removed, indexed with FAISS
over 20,000 training images.

| Metric | Value |
| --- | --- |
| Precision@5 | **92%** |
| Query latency | ~5 ms |
| Embedding | 384-d, L2-normalised, cosine similarity |

Retrieves **semantically** similar images rather than near-duplicates —
classification features collapse intra-class variation by construction. For
near-duplicate detection a perceptual hash would be the right tool.

### Model validation

`models/validation/` provides drift detection, A/B testing, and regression
gates, all operating on the inference audit trail.

```bash
uv run python scripts/analyse_drift.py --baseline-days 7 --current-days 1
uv run python scripts/check_regression.py
```

Drift uses PSI for categorical features and Kolmogorov-Smirnov for continuous
ones, with **severity driven by effect size rather than p-value** — at
production volumes a hypothesis test reports permanent drift. A/B comparisons
likewise require a difference to be both statistically significant *and*
materially large before it blocks a rollout.

### Inference performance

ViT-Small/16 at 224x224, batch 32, RTX 5000 Ada. Full matrix in
[`benchmarks/README.md`](benchmarks/README.md); analysis and deployment
recommendation in [`benchmarks/ANALYSIS.md`](benchmarks/ANALYSIS.md).

| Backend | Latency | Throughput | Speedup | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| PyTorch eager FP32 | 16.78 ms | 1,907 img/s | 1.00x | 87.25% |
| PyTorch eager bf16 | 5.23 ms | 6,118 img/s | 3.21x | 87.25% |
| **TensorRT FP16** | **3.49 ms** | **9,177 img/s** | **4.81x** | 87.25% |
| ONNX Runtime CPU INT8 | 475 ms | 67 img/s | 0.04x | 81.85% |

Single-image latency is 1.05 ms under TensorRT, three orders of magnitude
inside the sub-second requirement.

**INT8 is not recommended for this model.** It costs 5.4 points of top-1
accuracy for a 1.3x CPU speedup. Getting even that required replacing ONNX
Runtime's default min-max calibration, which cost 17.7 points, with percentile
calibration — vision transformers produce rare LayerNorm and GELU outliers that
destroy a min-max quantization range. See the analysis for the full calibration
study.

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

Six endpoints under `/api/v1`, plus unprefixed probes for orchestrators.

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

**243 tests, 95.06% statement coverage** of `api/`.

| Suite | Count | Scope |
| --- | ---: | --- |
| Unit | 199 | Validation, preprocessing, auth, rate limiting, cache, model loading |
| Integration | 44 | Full request path with a fake Redis and a synthetic model |
| Performance | 9 | Latency, memory stability, concurrency, batching |

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
the serving preprocessing matches the training transform to 7.2e-07 across eight
input shapes. It exists because two real bugs were found that way, both silent:
`v2.Resize` defaults to BILINEAR rather than BICUBIC, and `CenterCrop` rounds
rather than floor-divides. Since the serving path deliberately reimplements
preprocessing in NumPy so the production image needs no torch, that test is the
only thing preventing the two implementations from drifting apart again.

## Project layout

```
api/          FastAPI application (routers, services, middleware, schemas)
worker/       Celery application and batch inference tasks
ml/           Dataset pipelines, training, optimisation, model validation
worker/       Celery tasks for batch inference
tests/        Unit, integration, and performance suites
monitoring/   Prometheus scrape config and Grafana dashboards
docker/       Dockerfiles and nginx gateway configuration
scripts/      Dataset download and operational utilities
benchmarks/   Generated performance comparison reports
docs/         Model cards and the technical write-up
```

## Notes on the provided starter scripts

The challenge repository ships helper scripts under `scripts/`. Two contain
defects that would corrupt results if used as-is, so this project supplies
corrected implementations under `ml/` and documents the originals here.

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
becomes meaningless. `ml/data/` restructures the validation split into proper
per-class directories before loading.

## Documentation

| Document | Contents |
| --- | --- |
| [Technical write-up](docs/technical-writeup.md) | Model selection, optimisation results, architecture decisions, scalability, and an explicit list of gaps |
| [Model card: classifier](docs/model-card-classifier.md) | Metrics, training procedure, limitations, ethical considerations |
| [Model card: detector](docs/model-card-detector.md) | RT-DETR provenance, licensing rationale, limitations |
| [Benchmark results](benchmarks/README.md) | Generated latency matrix across backends |
| [Benchmark analysis](benchmarks/ANALYSIS.md) | Interpretation and deployment recommendation |
| [OpenAPI spec](docs/openapi.json) | Exported schema; also served live at `/openapi.json` |

Four silent bugs found during development — two preprocessing defects, a
backend that created a working session but could not infer, and a metrics
mislabelling that made every dashboard useless — are written up in
[section 4](docs/technical-writeup.md#4-four-bugs-worth-reporting).

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs four jobs: lint and
type-check; tests against real Postgres and Redis services with a 90% coverage
gate and a migration up/down/up cycle; a dependency and secret scan; and a
Docker build that asserts the image runs as non-root, stays under 1.5 GB,
contains no torch, and that the production Compose overlay refuses to render
without its secrets.

## Licence

MIT. Third-party model weights retain their upstream licences; see the model
cards in [`docs/`](docs/).
