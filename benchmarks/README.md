# Inference Performance Benchmarks

Generated: 2026-09-14T18:13:08+00:00

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
| classifier | onnxruntime-tensorrt | fp16 | 1 | 1.05 | 1.05 | 1.05 | 1.05 | 0.00 | 951.1 | 1.05 | 1.26x | 87.9 |
| classifier | pytorch-cuda-compiled | fp32 | 1 | 1.19 | 1.19 | 1.20 | 1.23 | 0.01 | 842.6 | 1.19 | 1.12x | - |
| classifier | onnxruntime-cuda | fp32 | 1 | 1.29 | 1.29 | 1.30 | 1.30 | 0.01 | 776.7 | 1.29 | 1.03x | 87.9 |
| classifier | pytorch-cuda | fp32 | 1 | 1.33 | 1.33 | 1.35 | 1.41 | 0.01 | 752.0 | 1.33 | 1.00x | - |
| classifier | pytorch-cuda | bf16 | 1 | 1.40 | 1.39 | 1.53 | 1.69 | 0.06 | 712.3 | 1.40 | 0.95x | - |
| classifier | onnxruntime-cpu | int8 | 1 | 13.14 | 11.96 | 19.08 | 31.59 | 3.30 | 76.1 | 13.14 | 0.10x | 23.1 |
| classifier | pytorch-cpu | fp32 | 1 | 14.61 | 14.58 | 14.95 | 15.37 | 0.19 | 68.5 | 14.61 | 0.09x | - |
| classifier | onnxruntime-cpu | fp32 | 1 | 15.01 | 14.20 | 19.23 | 34.58 | 3.62 | 66.6 | 15.01 | 0.09x | 87.9 |
| classifier | onnxruntime-tensorrt | fp16 | 8 | 1.37 | 1.37 | 1.37 | 1.37 | 0.00 | 5847.5 | 0.17 | 3.22x | 87.9 |
| classifier | pytorch-cuda | bf16 | 8 | 1.64 | 1.63 | 1.68 | 1.75 | 0.03 | 4888.4 | 0.20 | 2.69x | - |
| classifier | onnxruntime-cuda | fp32 | 8 | 3.48 | 3.48 | 3.50 | 3.52 | 0.01 | 2299.5 | 0.43 | 1.27x | 87.9 |
| classifier | pytorch-cuda-compiled | fp32 | 8 | 4.14 | 4.14 | 4.19 | 4.24 | 0.02 | 1931.1 | 0.52 | 1.06x | - |
| classifier | pytorch-cuda | fp32 | 8 | 4.41 | 4.41 | 4.43 | 4.43 | 0.02 | 1815.8 | 0.55 | 1.00x | - |
| classifier | pytorch-cpu | fp32 | 8 | 69.88 | 69.78 | 70.55 | 71.91 | 0.51 | 114.5 | 8.74 | 0.06x | - |
| classifier | onnxruntime-cpu | int8 | 8 | 91.74 | 88.66 | 111.59 | 156.65 | 12.45 | 87.2 | 11.47 | 0.05x | 23.1 |
| classifier | onnxruntime-cpu | fp32 | 8 | 116.10 | 114.03 | 149.21 | 173.01 | 15.35 | 68.9 | 14.51 | 0.04x | 87.9 |
| classifier | onnxruntime-tensorrt | fp16 | 32 | 3.63 | 3.62 | 3.74 | 3.78 | 0.08 | 8821.6 | 0.11 | 4.60x | 87.9 |
| classifier | pytorch-cuda | bf16 | 32 | 5.18 | 5.19 | 5.21 | 5.22 | 0.02 | 6175.9 | 0.16 | 3.22x | - |
| classifier | pytorch-cuda-compiled | fp32 | 32 | 16.47 | 16.47 | 16.50 | 16.64 | 0.03 | 1942.8 | 0.51 | 1.01x | - |
| classifier | pytorch-cuda | fp32 | 32 | 16.69 | 16.68 | 16.73 | 16.75 | 0.02 | 1917.5 | 0.52 | 1.00x | - |
| classifier | onnxruntime-cuda | fp32 | 32 | 19.04 | 19.04 | 19.11 | 19.35 | 0.07 | 1680.4 | 0.60 | 0.88x | 87.9 |
| classifier | pytorch-cpu | fp32 | 32 | 294.43 | 293.32 | 305.02 | 305.53 | 3.84 | 108.7 | 9.20 | 0.06x | - |
| classifier | onnxruntime-cpu | int8 | 32 | 522.75 | 510.98 | 597.28 | 730.45 | 47.77 | 61.2 | 16.34 | 0.03x | 23.1 |
| classifier | onnxruntime-cpu | fp32 | 32 | 560.29 | 561.31 | 610.88 | 626.20 | 28.73 | 57.1 | 17.51 | 0.03x | 87.9 |
| detector | onnxruntime-tensorrt | fp16 | 1 | 1.36 | 1.36 | 1.37 | 1.37 | 0.00 | 734.5 | 1.36 | - | 84.3 |
| detector | onnxruntime-cuda | fp32 | 1 | 3.91 | 3.90 | 3.98 | 4.00 | 0.05 | 256.0 | 3.91 | - | 84.3 |
| detector | onnxruntime-cpu | fp32 | 1 | 67.58 | 67.84 | 85.83 | 93.21 | 10.94 | 14.8 | 67.58 | - | 84.3 |
| detector | onnxruntime-tensorrt | fp16 | 8 | 8.26 | 8.28 | 8.34 | 8.37 | 0.06 | 968.0 | 1.03 | - | 84.3 |
| detector | onnxruntime-cuda | fp32 | 8 | 29.37 | 29.35 | 29.60 | 29.73 | 0.13 | 272.4 | 3.67 | - | 84.3 |
| detector | onnxruntime-cpu | fp32 | 8 | 557.63 | 551.35 | 605.65 | 675.35 | 29.86 | 14.3 | 69.70 | - | 84.3 |
| embedder | onnxruntime-tensorrt | fp16 | 1 | 0.52 | 0.52 | 0.52 | 0.52 | 0.00 | 1922.5 | 0.52 | - | 87.8 |
| embedder | onnxruntime-cuda | fp32 | 1 | 1.21 | 1.21 | 1.22 | 1.23 | 0.01 | 827.2 | 1.21 | - | 87.8 |
| embedder | onnxruntime-cpu | fp32 | 1 | 16.19 | 16.61 | 20.06 | 20.95 | 2.25 | 61.8 | 16.19 | - | 87.8 |
| embedder | onnxruntime-tensorrt | fp16 | 8 | 1.14 | 1.14 | 1.15 | 1.16 | 0.00 | 7020.9 | 0.14 | - | 87.8 |
| embedder | onnxruntime-cuda | fp32 | 8 | 3.55 | 3.54 | 3.59 | 3.63 | 0.02 | 2253.8 | 0.44 | - | 87.8 |
| embedder | onnxruntime-cpu | fp32 | 8 | 125.91 | 127.66 | 145.58 | 153.60 | 11.99 | 63.5 | 15.74 | - | 87.8 |
| embedder | onnxruntime-tensorrt | fp16 | 32 | 3.55 | 3.55 | 3.63 | 3.65 | 0.04 | 9005.6 | 0.11 | - | 87.8 |
| embedder | onnxruntime-cuda | fp32 | 32 | 18.98 | 18.98 | 19.03 | 19.07 | 0.03 | 1686.0 | 0.59 | - | 87.8 |
| embedder | onnxruntime-cpu | fp32 | 32 | 540.20 | 541.74 | 587.85 | 628.62 | 23.58 | 59.2 | 16.88 | - | 87.8 |
