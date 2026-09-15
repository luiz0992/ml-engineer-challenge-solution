# Memory profile

ONNX Runtime initialisation costs **839.5 MB** once, at first session creation, independently of which model triggers it. Measured separately below so the per-model figures are comparable.

Resident after initialisation: **909.9 MB**; **1303.0 MB** with all three models loaded.

## Resident cost per model

| Model | Backend | Provider | RSS delta |
| --- | --- | --- | ---: |
| classifier | onnx | CUDAExecutionProvider | 79.0 MB |
| detector | onnx | CUDAExecutionProvider | 226.5 MB |
| embedder | onnx | CUDAExecutionProvider | 87.6 MB |

## Peak allocation per classification

Averaged over 50 requests, 512x512 input.

| Statistic | Peak allocation |
| --- | ---: |
| mean | 1.791 MB |
| p50 | 1.793 MB |
| p95 | 1.802 MB |
| max | 1.803 MB |

RSS growth across the run: **0.2 MB**.

## What each request retains

Allocations still live after the request completed -- where a leak would appear. The transient peak above is dominated by decode and resize buffers that are freed before the response is sent.

| Location | Retained | Blocks |
| --- | ---: | ---: |
| `inference_service.py:296` | 0.2 KB | 3 |
| `inference_service.py:157` | 0.1 KB | 2 |
| `validators.py:112` | 0.1 KB | 1 |
| `inference_service.py:303` | 0.1 KB | 4 |
| `inference_service.py:299` | 0.1 KB | 1 |

## Scaling with upload size

| Input | Upload | Peak allocation |
| --- | ---: | ---: |
| 224x224 | 44.5 KB | 1.764 MB |
| 512x512 | 230.0 KB | 1.765 MB |
| 1024x1024 | 918.7 KB | 1.765 MB |
| 2048x2048 | 3671.5 KB | 1.765 MB |
