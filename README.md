# Multi-Model Computer Vision API

A production-oriented MLOps system that serves three computer-vision models —
image classification, object detection, and image similarity search — behind a
single authenticated, rate-limited, observable HTTP API.

> **Status:** in development. Sections marked _TBD_ are filled in as each layer
> lands; see [Project layout](#project-layout) for what currently exists.

---

## Contents

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Notes on the provided starter scripts](#notes-on-the-provided-starter-scripts)
- [Design decisions](#design-decisions)

---

## Architecture

_TBD — added in Layer 7 alongside the Compose stack._

## Quick start

Requires Docker with the NVIDIA container runtime (for GPU inference) and
[uv](https://docs.astral.sh/uv/) for local development.

```bash
cp .env.example .env          # then edit; JWT_SECRET_KEY must be set
uv sync --extra dev           # serving + test dependencies
uv sync --extra train         # adds torch/CUDA, only needed for training
```

## Project layout

```
api/          FastAPI application (routers, services, middleware, schemas)
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

## Design decisions

Rationale for model selection, optimisation strategy, and system architecture
lives in [`docs/technical-writeup.md`](docs/technical-writeup.md). _TBD._

## Licence

MIT. Third-party model weights retain their upstream licences; see the model
cards in [`docs/`](docs/).
