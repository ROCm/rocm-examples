# AMD ROCm Programming Guide: Tiling Matrix Multiply

## Description

This tutorial demonstrates matrix multiplication using various tiling
optimization techniques in HIP. It implements three different kernels
to illustrate the performance benefits of using shared memory (LDS)
tiling and register tiling for matrix operations.

The example performs matrix multiplication: $C = A \times B$ where $A$
is an $M \times K$ matrix, $B$ is a $K \times N$ matrix, and $C$ is the
resulting $M \times N$ matrix.

### Application flow

1. Device capabilities are queried to determine the warp size.
2. Random input matrices $A$ and $B$ are generated on the host. Matrix
   $B$ is set to the identity matrix for easy verification.
3. Device memory is allocated and input matrices are copied to the
   device.
4. Three different matrix multiplication kernels are executed and
   verified:
   - Naive implementation without tiling
   - LDS (Local Data Share / shared memory) tiling implementation
   - Register tiling implementation with advanced optimization
5. Each kernel's result is verified against the expected output.
6. Device memory is freed.

## Key APIs and Concepts

- **Naive kernel**: Each thread computes one output element by
  iterating through the corresponding row and column, with no data
  reuse optimization.
- **LDS tiling kernel**: Uses shared memory tiles to cache portions of
  the input matrices, reducing global memory accesses. The
  `base_tile_size` parameter (16x16) defines the tile dimensions.
  - `__shared__` declares shared memory accessible by all threads in a
    block.
  - `__syncthreads()` synchronizes all threads in a block before
    accessing shared data.
- **Register tiling kernel**: Advanced optimization where each thread
  computes a tile of output elements (4x4) using register blocking.
  Combines LDS tiling with register-level data reuse.
  - Uses warp-level tiling for better memory coalescing.
  - Computes multiple output elements per thread to maximize register
    utilization.
- `hipDeviceGetAttribute` queries device properties such as warp size.
- `hipGetLastError` checks for errors from the last HIP operation.
- `hipDeviceSynchronize` blocks the host until all queued device
  operations complete.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`

#### Host symbols

- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
- `hipGetLastError`
- `hipDeviceSynchronize`
- `hipDeviceGetAttribute`
- `hipDeviceAttributeWarpSize`

#### Device functions

- `__syncthreads()`
