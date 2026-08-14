# hipThreads Sparse Matrix Multiplication Examples

## Description

This series implements sparse matrix-matrix multiplication (SpMM): given two sparse matrices A (in CSR format) and B (in CSC format) read from MatrixMarket files, it computes C = A × B. The series starts from a single-threaded CPU baseline and progressively adds parallelism and moves computation to the AMD GPU using hipThreads. Sample matrices are provided in the `data/` directory.

### Steps

| Step | Directory | Description |
|------|-----------|-------------|
| 1 | `step1_baseline/` | Single-threaded CPU baseline. |
| 2 | `step2_cpu_threading/` | CPU threading: `std::thread` partitions output rows across threads. |
| 2.5 | `step2_5_gpu_data_structures/` | GPU data structures: matrix data moved to GPU-resident memory via rocThrust, computation still on the CPU. |
| 3 | `step3_hipthread_port/` | GPU port: `hip::wthread` runs the multiplication kernel on the GPU. |
