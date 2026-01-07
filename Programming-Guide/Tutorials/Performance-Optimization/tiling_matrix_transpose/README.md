# AMD ROCm Programming Guide: Tiling Matrix Transpose

## Description

This tutorial demonstrates matrix transposition using tiling techniques
in HIP. It implements two different kernels to illustrate the
performance benefits of using shared memory (LDS) tiling to improve
memory access patterns and achieve better memory coalescing.

Matrix transposition converts an $M \times N$ matrix into an
$N \times M$ matrix by swapping rows and columns: $B[j][i] = A[i][j]$.

### Application flow

1. Input and output matrices are allocated on the host with dimensions
   8192x8192.
2. The input matrix is initialized with random floating-point values.
3. Device memory is allocated for input and output matrices.
4. The input matrix is copied to device memory.
5. Two different matrix transpose kernels are executed:
   - Naive implementation with direct global memory access
   - LDS tiling implementation for improved memory coalescing
6. The transposed matrix is copied back to host memory.
7. Results can be verified (verification code is commented out).
8. Device memory is freed.

## Key APIs and Concepts

- **Naive kernel** (`transpose_naive_kernel`): Each thread reads from
  one location and writes to another in global memory. This can lead to
  uncoalesced memory accesses, particularly for the write operation.
- **LDS tiling kernel** (`transpose_lds_kernel`): Uses shared memory
  tiles to improve memory access patterns:
  - Threads cooperatively load a tile of the input matrix into shared
    memory with coalesced reads.
  - `__syncthreads()` ensures all threads complete loading before
    proceeding.
  - Threads read from shared memory and write to global memory with
    coalesced writes.
  - The tile size (16x16) balances shared memory usage with memory
    coalescing benefits.
- Memory coalescing: Accessing consecutive memory locations by
  consecutive threads improves memory bandwidth utilization.
- `hipMalloc` allocates device memory.
- `hipMemcpy` transfers data between host and device.
- `hipDeviceSynchronize` ensures all device operations complete before
  continuing.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`

#### Host symbols

- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
- `hipDeviceSynchronize`

#### Device functions

- `__syncthreads()`
