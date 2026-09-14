# Inference Performance Benchmarks

Generated: 2026-09-10T20:07:59+00:00

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

| Backend | Precision | Batch | Mean (ms) | p50 | p95 | p99 | Std | Throughput (img/s) | Per-image (ms) | Speedup | Size (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| onnxruntime-tensorrt | fp16 | 1 | 1.05 | 1.05 | 1.05 | 1.06 | 0.00 | 951.4 | 1.05 | 1.26x | 87.9 |
| pytorch-cuda-compiled | fp32 | 1 | 1.19 | 1.18 | 1.19 | 1.27 | 0.01 | 843.3 | 1.19 | 1.11x | - |
| onnxruntime-cuda | fp32 | 1 | 1.28 | 1.28 | 1.30 | 1.34 | 0.01 | 780.1 | 1.28 | 1.03x | 87.9 |
| pytorch-cuda | fp32 | 1 | 1.32 | 1.32 | 1.34 | 1.40 | 0.01 | 756.7 | 1.32 | 1.00x | - |
| pytorch-cuda | bf16 | 1 | 1.38 | 1.38 | 1.40 | 1.43 | 0.01 | 723.7 | 1.38 | 0.96x | - |
| onnxruntime-cpu | int8 | 1 | 11.19 | 10.89 | 13.52 | 13.93 | 1.05 | 89.4 | 11.19 | 0.12x | 23.1 |
| pytorch-cpu | fp32 | 1 | 14.04 | 14.03 | 15.31 | 16.95 | 0.67 | 71.2 | 14.04 | 0.09x | - |
| onnxruntime-cpu | fp32 | 1 | 16.55 | 16.17 | 20.87 | 24.75 | 2.97 | 60.4 | 16.55 | 0.08x | 87.9 |
| onnxruntime-tensorrt | fp16 | 8 | 1.37 | 1.37 | 1.39 | 1.39 | 0.01 | 5821.1 | 0.17 | 3.21x | 87.9 |
| pytorch-cuda | bf16 | 8 | 1.62 | 1.61 | 1.68 | 1.70 | 0.02 | 4930.1 | 0.20 | 2.72x | - |
| onnxruntime-cuda | fp32 | 8 | 3.51 | 3.48 | 3.67 | 3.92 | 0.08 | 2281.3 | 0.44 | 1.26x | 87.9 |
| pytorch-cuda-compiled | fp32 | 8 | 4.13 | 4.13 | 4.17 | 4.19 | 0.02 | 1937.7 | 0.52 | 1.07x | - |
| pytorch-cuda | fp32 | 8 | 4.41 | 4.41 | 4.43 | 4.45 | 0.01 | 1812.8 | 0.55 | 1.00x | - |
| pytorch-cpu | fp32 | 8 | 68.69 | 68.64 | 69.26 | 72.19 | 0.51 | 116.5 | 8.59 | 0.06x | - |
| onnxruntime-cpu | int8 | 8 | 87.03 | 81.09 | 125.70 | 177.52 | 17.51 | 91.9 | 10.88 | 0.05x | 23.1 |
| onnxruntime-cpu | fp32 | 8 | 109.71 | 107.11 | 139.02 | 175.62 | 13.04 | 72.9 | 13.71 | 0.04x | 87.9 |
| onnxruntime-tensorrt | fp16 | 32 | 3.49 | 3.48 | 3.59 | 3.68 | 0.05 | 9176.6 | 0.11 | 4.81x | 87.9 |
| pytorch-cuda | bf16 | 32 | 5.23 | 5.23 | 5.24 | 5.25 | 0.01 | 6118.2 | 0.16 | 3.21x | - |
| pytorch-cuda | fp32 | 32 | 16.78 | 16.79 | 16.81 | 16.84 | 0.02 | 1906.7 | 0.52 | 1.00x | - |
| pytorch-cuda-compiled | fp32 | 32 | 16.80 | 16.79 | 16.93 | 17.09 | 0.07 | 1904.9 | 0.52 | 1.00x | - |
| onnxruntime-cuda | fp32 | 32 | 18.86 | 18.85 | 18.96 | 19.25 | 0.06 | 1696.3 | 0.59 | 0.89x | 87.9 |
| pytorch-cpu | fp32 | 32 | 295.03 | 293.38 | 310.39 | 324.91 | 6.07 | 108.5 | 9.22 | 0.06x | - |
| onnxruntime-cpu | int8 | 32 | 475.32 | 456.06 | 644.27 | 881.86 | 78.71 | 67.3 | 14.85 | 0.04x | 23.1 |
| onnxruntime-cpu | fp32 | 32 | 617.27 | 567.47 | 949.24 | 1135.97 | 141.67 | 51.8 | 19.29 | 0.03x | 87.9 |
