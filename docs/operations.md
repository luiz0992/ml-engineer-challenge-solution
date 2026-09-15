# Operations runbook

What to do, rather than why it was built that way. Design rationale lives in
the [technical write-up](technical-writeup.md); this is the document you want
at 03:00.

- [Deploying](#deploying)
- [Scaling](#scaling)
- [Shipping a new model version](#shipping-a-new-model-version)
- [Running an A/B experiment](#running-an-ab-experiment)
- [Scheduled jobs](#scheduled-jobs)
- [Alert runbook](#alert-runbook)
- [Troubleshooting](#troubleshooting)
- [Rollback](#rollback)

---

## Deploying

### Local

```bash
docker compose up -d
curl localhost:8080/api/v1/health
```

Requires `JWT_SECRET_KEY` and `POSTGRES_PASSWORD`; copy `.env.example` to
`.env` first. The stack seeds one API key per tier (`dev-key-free`,
`dev-key-pro`, `dev-key-enterprise`) — **only** when `APP_ENV` is not
`production`.

### Production overlay

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

The overlay refuses to render without `REDIS_PASSWORD` and `GRAFANA_PASSWORD`
in addition to the two above. That is deliberate: a deploy that silently falls
back to a development credential is the failure the `${VAR:?}` syntax exists to
prevent, and CI asserts the refusal.

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Adds device reservations and `onnxruntime-gpu`. Separate overlay because a
Compose file demanding a GPU fails outright on a machine without one. Takes
single-image classification from 15.9 ms to about 1 ms.

### Alert delivery

```bash
docker compose -f docker-compose.yml -f docker-compose.alerts.yml up -d
```

Points every Alertmanager receiver at the in-stack webhook sink
(`docker/webhook`), which writes alerts to JSONL. The base config has no
endpoints because an unset Slack URL crashes Alertmanager. Production
webhooks still belong in a secret manager.

### Kubernetes

```bash
kubectl apply -f deploy/kubernetes/
```

Ten resources, validated against real 1.30 API schemas with `kubeconform
--strict` and applied to a kind cluster in CI. Pods will not become ready
there without images and a PVC backend; the apply itself is what the job
asserts.

---

## Scaling

### What the autoscaler targets

`http_requests_in_progress` per pod, not CPU. Inference saturates the GPU — or
the thread pool that ONNX Runtime's blocking `Run()` occupies — well before CPU
utilisation looks high, so a CPU-targeted autoscaler scales *after* latency has
already degraded. CPU remains a secondary guard at 80%, which catches a
genuinely CPU-bound regression such as a silent fallback to the CPU execution
provider.

Scale-down is deliberately slow (one pod per minute, five-minute
stabilisation): a terminating pod discards its in-memory model and the next one
pays the load-and-warm cost again.

### Measured capacity

One replica, 4-CPU quota, CPU inference, from
[`benchmarks/load-test.md`](../benchmarks/load-test.md):

| | p50 | p95 |
| --- | ---: | ---: |
| `/classify` | 23 ms | 67 ms |
| `/classify` (cache hit) | 3 ms | 14 ms |
| `/detect` | 96 ms | 240 ms |

Zero failures at 20 concurrent users and ~15 req/s. The service was not
saturated, so treat these as a floor rather than a ceiling.

### What to scale first

1. **`ml-api` replicas.** Stateless; scale freely. This is almost always the
   right first move.
2. **`worker` concurrency.** Capped at 2 because each process loads its own
   copy of the models — 1.30 GB resident on a GPU host, of which 840 MB is the
   runtime rather than the weights ([`memory.md`](../benchmarks/memory.md)).
   Raising concurrency multiplies that; add worker *replicas* instead.
3. **Separate Redis.** Cache, rate limits, and the Celery broker share one
   instance. Split them before scaling far.

**Always set a CPU limit.** ONNX Runtime sizes its thread pool from the CPU
budget it can see, and `available_cpus()` reads the cgroup quota to size it
correctly. An unlimited container on a many-core host behaves very differently
from a production one — see [Troubleshooting](#inference-is-mysteriously-slow-in-a-container).

---

## Shipping a new model version

Versioning is resolved from the artefact layout, so a second version is a
deployment step rather than a code change.

```
models/artifacts/onnx/
├── classifier_fp32.onnx          # v1, flat layout
├── labels.json                   # (at the artifacts root)
└── v2/
    ├── classifier_fp32.onnx      # v2 weights
    └── labels.json               # and v2's own class list
```

1. Train and publish a second version. The artefacts land under
   `onnx/<version>/`, **including that version's `labels.json`**. A non-default
   version must be self-describing: without its own label map it is refused at
   load time rather than named from another version's class list.

   ```bash
   uv run python -m models.training.train --config-name train_classifier_v2
   uv run python scripts/prepare_artifacts.py \
       --run-dir models/artifacts/runs/classifier-v2 --version v2
   ```
2. Restart `ml-api`. Every version on disk is loaded at startup and logged as
   `model_loaded`; the extras are registered but **not** made active.
3. Verify it serves, without moving any traffic:

   ```bash
   curl -X POST "localhost:8080/api/v1/classify?model_version=v2" \
        -H "Authorization: Bearer $TOKEN" -F "file=@image.jpg"
   ```

4. Move traffic with an experiment (below), or promote it outright.

A pinned version that is not on disk fails at load time rather than silently
serving `v1`. That restriction matters: a fallback would mean the response, the
provenance block, the audit row, and the A/B analysis all agreeing on a version
that never ran.

---

## Running an A/B experiment

Write `models/artifacts/experiments.json`:

```json
{
  "experiments": [
    {
      "name": "rollout",
      "model_name": "tiny-imagenet-classifier",
      "enabled": true,
      "variants": [
        {"name": "control",   "model_version": "v1", "weight": 0.9},
        {"name": "treatment", "model_version": "v2", "weight": 0.1}
      ]
    }
  ]
}
```

Restart `ml-api`. Assignment is deterministic per user (SHA-256 of
`experiment + user_id`), so a user stays on their arm.

**A variant naming an unloaded version is rejected at startup**, rather than
404-ing that share of traffic. Weights that do not sum to 1.0, duplicate
variant names, and malformed JSON all disable the experiment with a loud log
and send everything to the active version.

Analyse from the audit trail — every row carries the `variant` that served it:

```sql
SELECT variant, model_version, count(*), avg(latency_ms), 
       avg((status = 'success')::int) AS success_rate
FROM inference_logs
WHERE variant IS NOT NULL AND created_at > now() - interval '7 days'
GROUP BY variant, model_version;
```

`models/validation/ab_testing.py` provides Welch's t-test, a two-proportion
z-test, and a recommendation that requires a difference to be both
statistically significant *and* materially large.

---

## Scheduled jobs

| Job | Image | Schedule | What it does |
| --- | --- | --- | --- |
| `migrate` | `mlchal-api` | once, at startup | Applies Alembic migrations, then exits |
| `maintenance` | `mlchal-jobs` | daily | Creates next month's audit partitions, drops expired ones (6-month retention) |
| `drift` | `mlchal-jobs` | daily | PSI and KS tests over the audit trail, pushes gauges to Pushgateway |

Run one by hand. These services set a shell entrypoint for their scheduling
loop, so override it:

```bash
docker compose run --entrypoint python maintenance -m scripts.maintain_partitions
docker compose run --entrypoint python drift -m scripts.analyse_drift \
    --baseline-days 7 --current-days 1
```

Both push to Pushgateway rather than being scraped, because a batch job has
exited by the time Prometheus comes looking.

**They run on `mlchal-jobs`, not the API image.** `analyse_drift` runs a
Kolmogorov-Smirnov test and needs scipy, which is deliberately kept out of the
serving layer — it would add ~160 MB to every API and worker replica for a
dependency only a nightly batch job uses (701 MB against the jobs image's
863 MB). Sharing the API image meant the drift
job crashed with `ModuleNotFoundError` the first time both windows had enough
data to reach the KS test; before that it exited early on an empty baseline and
looked like it was working.

Drift severity is driven by **effect size, not p-value**. A real run on this
stack reported `latency_ms` with `p=1.83e-70` and severity `none`, because the
KS statistic was only 0.0866. At production volumes a hypothesis test reports
permanent drift, which is why the threshold sits on the statistic.

---

## Alert runbook

Ten rules in `monitoring/prometheus/alerts.yml`. **Model-quality alerts are
deliberately not paged** — drift is not an incident, it is a signal that the
model's assumptions are expiring, and paging on it is the fastest route to
having it muted permanently.

### Critical

| Alert | Means | First actions |
| --- | --- | --- |
| `APIDown` | No scrape response for 1 min | Check `docker compose ps` / pod status. Startup is fail-fast on the model: if the classifier cannot load, the process exits by design. Check logs for `model_load_failed`. |
| `NoModelLoaded` | Running with zero models | Should be unreachable — startup fails without a classifier. Indicates the artefact volume vanished under a running process. Check the `model-artifacts` mount. |
| `HighErrorRate` | >5% of requests 5xx for 5 min | Check `/api/v1/health` for the degraded component. Inference errors and dependency errors look different: the former shows in `inference_requests_total{status="failure"}`. |

### Warning

| Alert | Means | First actions |
| --- | --- | --- |
| `ModelRunningDegraded` | Serving on a fallback backend | The preferred backend failed its warmup probe. Grep for `backend_unavailable`. Common cause is a missing CUDA/TensorRT library after a base-image change. Traffic is being served — this is not urgent, but it is usually a 10x latency regression. |
| `InferenceFailures` | >1% of inferences failing | Distinct from `HighErrorRate`: the request reached a model and the model failed. Check for malformed inputs getting past validation. |
| `HighRequestLatency` | p95 >500 ms for 10 min | Check `SaturatedRequestQueue` first — if both fire, it is load, so scale out. If latency is high with a shallow queue, suspect a degraded backend or a CPU-limit misconfiguration. |
| `SaturatedRequestQueue` | >100 requests in flight | The autoscaler's own signal. If it is not scaling, check the HPA and the metrics adapter. |
| `ModelDriftSignificant` | PSI or KS crossed the threshold for 1 h | **Do not page.** Investigate during working hours: `benchmarks/` and `scripts/analyse_drift.py --window` for the affected feature. Severity is driven by effect size, not p-value, so this means a materially different input distribution. |
| `ModelDriftJobStale` | No drift report in 48 h | The job is failing silently. Check `docker compose logs drift`. An alert on a metric nothing exports is permanently green and looks like coverage — this rule exists because that happened. |
| `AuditDefaultPartitionNonEmpty` | Rows landing in the `DEFAULT` partition | The maintenance job has not created the current month's partition. Rows are not lost, but they escape retention. Run `maintain_partitions.py` by hand, then check why the job stopped. |

---

## Troubleshooting

### Inference is mysteriously slow in a container

Almost certainly the CPU budget. `os.cpu_count()` reports the *host's*
processors inside a container, so ONNX Runtime builds a thread pool sized for
hardware the container cannot use and thrashes inside its quota. Measured on
this image: **102.8 ms per classification with the host count against 12.4 ms
with a pool matching a 4-CPU quota.**

`api/services/runtime.py::available_cpus` reads the cgroup quota, so this is
handled — but confirm the container actually has a limit set. An *unlimited*
container on a shared 64-core host will oversubscribe in exactly the same way.

```bash
docker compose exec ml-api cat /sys/fs/cgroup/cpu.max   # "<quota> <period>"
docker compose logs ml-api | grep onnx_threads_configured
```

### `/api/v1/metrics` returns 404 through the gateway

Working as intended. Metrics reveal traffic volumes, model names, and error
rates, so nginx blocks them at the edge; Prometheus scrapes `ml-api` directly
over the internal network. To read them by hand:

```bash
docker compose exec ml-api wget -qO- http://127.0.0.1:8000/api/v1/metrics
```

### Health says `degraded`

Expected when Redis or Postgres is unavailable — both fail open. Inference
still works; caching, rate limiting, or audit logging is off. `/health/ready`
is what the load balancer should use, and it stays ready. Only the loss of
inference itself is `unhealthy`.

### Rate limits are hit sooner than expected

Limits are per `user_id`, not per connection. All three seeded development keys
map to one user each, so twenty clients sharing `dev-key-pro` share one bucket
of 120/min. Issue distinct keys for load testing.

### The worker processes nothing

Celery needs Redis as a broker. Check `docker compose logs worker` for
connection errors, and confirm `redis` is healthy. Submitted jobs stay
`PENDING` rather than failing, which looks like a hang.

---

## Rollback

**Application.** Re-deploy the previous image tag. The API is stateless; no
coordination is needed.

**Model.** If a version was promoted and is misbehaving, promote the previous
one — both remain loaded. This does not require a restart if you promote
through the registry, and is instant.

**Database.** Every migration is reversible, and CI asserts
`upgrade → downgrade → upgrade` on every run. Roll back one step with:

```bash
docker compose run --rm migrate alembic downgrade -1
```

Check what a downgrade discards before running it in production: the
partitioning migration drops partitions, and dropping a partition drops its
rows.
