# hipThreads SAXPY Examples

## Description

This series implements a compute-bound SAXPY (Single-precision A·X + Y) benchmark. Each element undergoes 512 iterations of the expression `t = a*t + y[i]`, making the workload deliberately compute-bound rather than memory-bound. The series starts from a CPU baseline using `std::thread` and progressively ports the computation to the AMD GPU using hipThreads.

### Steps

| Step | Directory | Description |
|------|-----------|-------------|
| 1 | `step1_baseline/` | CPU baseline: `std::thread` partitions the element array across threads. |
| 2 | `step2_hipthread_dropin/` | GPU port: `std::thread` replaced with `hip::wthread` as a near drop-in; arrays moved to GPU-resident memory. |
| 3 | `step3_simdize/` | Wavefront SIMD: each `hip::wthread` is widened to fill a full wavefront, with each fiber processing a strided element. |
