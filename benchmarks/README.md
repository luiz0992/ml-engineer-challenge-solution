# Inference Performance Benchmarks

Generated: 2026-09-14T21:23:09+00:00

## Environment

| Property | Value |
| --- | --- |
| python | 3.12.11 |
| platform | Linux-7.0.0-30-generic-x86_64-with-glibc2.39 |
| processor | x86_64 |
| torch | 2.11.0+cu128 |
| cuda_available | True |
| gpu | NVIDIA RTX 5000 Ada Generation |
| cuda | 12.8 |
| compute_capability | sm_89 |
| gpu_memory_gb | 31.4 |
| onnxruntime | 1.29.0 |
| onnxruntime_providers | TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider |
| tensorrt | 10.16.1.11 |

## Latency

All timings exclude warmup and synchronise the device before stopping the clock. Speedup is relative to `pytorch-cuda` (fp32) at the same batch size.

| Model | Backend | Precision | Batch | Mean (ms) | p50 | p95 | p99 | Std | Throughput (img/s) | Per-image (ms) | Speedup | Size (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| classifier | onnxruntime-tensorrt | fp16 | 1 | 1.05 | 1.05 | 1.05 | 1.07 | 0.00 | 951.2 | 1.05 | 1.27x | 87.9 |
| classifier | pytorch-cuda-compiled | fp32 | 1 | 1.19 | 1.19 | 1.22 | 1.23 | 0.01 | 839.6 | 1.19 | 1.12x | - |
| classifier | onnxruntime-cuda | fp32 | 1 | 1.29 | 1.29 | 1.30 | 1.31 | 0.01 | 776.1 | 1.29 | 1.03x | 87.9 |
| classifier | pytorch-cuda | fp32 | 1 | 1.33 | 1.33 | 1.35 | 1.40 | 0.01 | 750.8 | 1.33 | 1.00x | - |
| classifier | pytorch-cuda | bf16 | 1 | 1.43 | 1.42 | 1.49 | 1.75 | 0.04 | 701.0 | 1.43 | 0.93x | - |
| classifier | onnxruntime-cpu | int8 | 1 | 10.69 | 10.34 | 12.13 | 13.47 | 0.70 | 93.6 | 10.69 | 0.12x | 23.1 |
| classifier | pytorch-cpu | fp32 | 1 | 12.32 | 12.30 | 12.56 | 12.77 | 0.11 | 81.2 | 12.32 | 0.11x | - |
| classifier | onnxruntime-cpu | fp32 | 1 | 17.71 | 17.66 | 23.57 | 25.00 | 3.21 | 56.5 | 17.71 | 0.08x | 87.9 |
| classifier | onnxruntime-tensorrt | fp16 | 8 | 1.37 | 1.37 | 1.38 | 1.40 | 0.01 | 5839.0 | 0.17 | 3.18x | 87.9 |
| classifier | pytorch-cuda | bf16 | 8 | 1.63 | 1.61 | 1.74 | 1.93 | 0.05 | 4906.4 | 0.20 | 2.67x | - |
| classifier | onnxruntime-cuda | fp32 | 8 | 3.48 | 3.48 | 3.51 | 3.52 | 0.02 | 2298.8 | 0.44 | 1.25x | 87.9 |
| classifier | pytorch-cuda-compiled | fp32 | 8 | 4.14 | 4.13 | 4.17 | 4.21 | 0.02 | 1934.0 | 0.52 | 1.05x | - |
| classifier | pytorch-cuda | fp32 | 8 | 4.36 | 4.36 | 4.39 | 4.43 | 0.02 | 1836.1 | 0.54 | 1.00x | - |
| classifier | pytorch-cpu | fp32 | 8 | 67.02 | 66.91 | 69.23 | 69.84 | 0.73 | 119.4 | 8.38 | 0.07x | - |
| classifier | onnxruntime-cpu | int8 | 8 | 87.83 | 86.30 | 100.23 | 102.93 | 5.90 | 91.1 | 10.98 | 0.05x | 23.1 |
| classifier | onnxruntime-cpu | fp32 | 8 | 106.86 | 106.49 | 121.10 | 127.22 | 7.02 | 74.9 | 13.36 | 0.04x | 87.9 |
| classifier | onnxruntime-tensorrt | fp16 | 32 | 3.51 | 3.51 | 3.62 | 3.67 | 0.05 | 9104.7 | 0.11 | 4.76x | 87.9 |
| classifier | pytorch-cuda | bf16 | 32 | 5.19 | 5.20 | 5.23 | 5.30 | 0.03 | 6161.6 | 0.16 | 3.22x | - |
| classifier | pytorch-cuda-compiled | fp32 | 32 | 16.57 | 16.57 | 16.65 | 16.81 | 0.04 | 1930.7 | 0.52 | 1.01x | - |
| classifier | pytorch-cuda | fp32 | 32 | 16.73 | 16.72 | 16.79 | 16.86 | 0.04 | 1913.3 | 0.52 | 1.00x | - |
| classifier | onnxruntime-cuda | fp32 | 32 | 18.94 | 18.94 | 19.05 | 19.13 | 0.05 | 1689.2 | 0.59 | 0.88x | 87.9 |
| classifier | pytorch-cpu | fp32 | 32 | 291.97 | 290.71 | 304.65 | 319.75 | 5.50 | 109.6 | 9.12 | 0.06x | - |
| classifier | onnxruntime-cpu | int8 | 32 | 460.80 | 460.86 | 504.75 | 527.39 | 24.56 | 69.4 | 14.40 | 0.04x | 23.1 |
| classifier | onnxruntime-cpu | fp32 | 32 | 519.26 | 517.89 | 555.57 | 570.21 | 18.35 | 61.6 | 16.23 | 0.03x | 87.9 |
| detector | onnxruntime-tensorrt | fp16 | 1 | 2.14 | 2.14 | 2.16 | 2.16 | 0.01 | 466.3 | 2.14 | - | 84.3 |
| detector | onnxruntime-cuda | fp32 | 1 | 3.71 | 3.69 | 3.87 | 3.88 | 0.07 | 269.6 | 3.71 | - | 84.3 |
| detector | onnxruntime-cpu | fp32 | 1 | 54.05 | 52.89 | 63.77 | 78.55 | 5.40 | 18.5 | 54.05 | - | 84.3 |
| detector | onnxruntime-cpu | int8 | 1 | 62.90 | 62.24 | 67.38 | 77.29 | 2.94 | 15.9 | 62.90 | - | 24.6 |
| detector | onnxruntime-tensorrt | fp16 | 8 | 8.29 | 8.27 | 8.51 | 8.75 | 0.10 | 965.3 | 1.04 | - | 84.3 |
| detector | onnxruntime-cuda | fp32 | 8 | 29.18 | 29.19 | 29.31 | 29.49 | 0.09 | 274.1 | 3.65 | - | 84.3 |
| detector | onnxruntime-cpu | int8 | 8 | 496.87 | 489.96 | 538.40 | 713.93 | 28.96 | 16.1 | 62.11 | - | 24.6 |
| detector | onnxruntime-cpu | fp32 | 8 | 537.43 | 535.33 | 581.79 | 601.55 | 20.48 | 14.9 | 67.18 | - | 84.3 |
| embedder | onnxruntime-tensorrt | fp16 | 1 | 1.06 | 1.06 | 1.06 | 1.06 | 0.00 | 947.5 | 1.06 | - | 87.8 |
| embedder | onnxruntime-cuda | fp32 | 1 | 1.30 | 1.30 | 1.30 | 1.31 | 0.01 | 771.9 | 1.30 | - | 87.8 |
| embedder | onnxruntime-cpu | int8 | 1 | 12.27 | 11.48 | 16.13 | 18.94 | 1.77 | 81.5 | 12.27 | - | 23.2 |
| embedder | onnxruntime-cpu | fp32 | 1 | 13.39 | 13.01 | 16.34 | 18.74 | 1.77 | 74.7 | 13.39 | - | 87.8 |
| embedder | onnxruntime-tensorrt | fp16 | 8 | 1.38 | 1.38 | 1.41 | 1.45 | 0.01 | 5789.2 | 0.17 | - | 87.8 |
| embedder | onnxruntime-cuda | fp32 | 8 | 3.48 | 3.48 | 3.51 | 3.57 | 0.02 | 2296.3 | 0.44 | - | 87.8 |
| embedder | onnxruntime-cpu | int8 | 8 | 95.55 | 93.15 | 117.81 | 154.86 | 10.56 | 83.7 | 11.94 | - | 23.2 |
| embedder | onnxruntime-cpu | fp32 | 8 | 107.87 | 107.06 | 124.71 | 154.97 | 9.27 | 74.2 | 13.48 | - | 87.8 |
| embedder | onnxruntime-tensorrt | fp16 | 32 | 3.48 | 3.48 | 3.55 | 3.57 | 0.04 | 9191.6 | 0.11 | - | 87.8 |
| embedder | onnxruntime-cuda | fp32 | 32 | 19.08 | 19.08 | 19.20 | 19.32 | 0.07 | 1677.0 | 0.60 | - | 87.8 |
| embedder | onnxruntime-cpu | int8 | 32 | 475.13 | 476.43 | 507.85 | 534.61 | 20.51 | 67.4 | 14.85 | - | 23.2 |
| embedder | onnxruntime-cpu | fp32 | 32 | 512.41 | 506.74 | 568.68 | 644.64 | 28.44 | 62.5 | 16.01 | - | 87.8 |

## Accuracy

| Backend | Precision | Top-1 | Notes |
| --- | --- | ---: | --- |
| onnxruntime-cpu | fp32 | 87.25% | top-5 97.50%, n=2000 |
| onnxruntime-cpu | int8 | 81.85% | top-5 94.80%, n=2000, -5.40pp vs FP32 |
| onnxruntime-cuda | fp32 | 87.25% | top-5 97.50%, n=2000 |
