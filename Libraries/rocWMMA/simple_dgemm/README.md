# rocWMMA Simple Double-Precision General Matrix Multiplication (DGEMM)

## Description

This example demonstrates a basic double-precision General Matrix Multiplication using rocWMMA. It showcases the library's support for FP64 computations on compatible AMD GPU architectures.

The operation calculates the following product:

$D = \alpha \cdot A \cdot B + \beta \cdot C$

where:

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$

## Application flow

1. **Device Capability Check**: The application begins by checking if the current device supports double-precision (FP64) operations using the `is_f64_supported()` utility function.
2. **Matrix Initialization**: Host-side matrices (A, B, C, D) are allocated and filled with random data.
3. **Device Memory Allocation**: Device memory is allocated for all matrices, and the input matrices (A, B, C) are copied from the host to the device.
4. **Kernel Launch**: The `dgemm_rocwmma_d` kernel is launched with a grid and block configuration calculated to cover the entire output matrix.
5. **GEMM Computation**: Inside the kernel, each warp:
    - Calculates its target output block in the D matrix.
    - Loops over the K dimension, loading tiles of A and B into rocWMMA fragments.
    - Performs matrix multiplication and accumulation using `rocwmma::mma_sync`.
    - Loads the corresponding tile of C, performs the final scaling with alpha and beta, and stores the result to D.
6. **Result Verification**: The output matrix D is copied back to the host and compared against a CPU-based reference implementation for validation.
7. **Cleanup**: All device memory is deallocated.

## Key APIs and Concepts

- **Double-Precision (FP64) Support**: This example specifically uses `float64_t` for all matrix data and computations. Support for FP64 is hardware-dependent and is checked at runtime.

- **rocWMMA Fragments**:
  - `rocwmma::fragment`: The core data structure in rocWMMA, representing a tile of a matrix held in a warp's registers. Fragments are defined for matrices A, B, and the accumulator (C/D).
  - `rocwmma::load_matrix_sync()`: Loads a tile of a matrix from global memory into a fragment. The function handles the complex indexing and memory access patterns.
  - `rocwmma::mma_sync()`: Executes the matrix multiplication and accumulation operation ($D = A \cdot B + C$) on the fragments.
  - `rocwmma::store_matrix_sync()`: Stores the data from an accumulator fragment back to global memory.
  - `rocwmma::fill_fragment()`: Initializes an accumulator fragment with a specific value (typically 0.0).

- **Kernel Tiling Strategy**:
  - The problem is tiled into smaller blocks that are processed by individual warps.
  - The kernel uses a simple 2D grid of thread blocks and maps each warp to a specific tile of the output matrix C/D.
  - The calculation for each tile involves iterating through the K dimension and accumulating the results.

## Demonstrated API Calls

### rocWMMA

- `rocwmma::fragment`
- `rocwmma::load_matrix_sync`
- `rocwmma::store_matrix_sync`
- `rocwmma::mma_sync`
- `rocwmma::fill_fragment`

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
- `hipGetDevice`
- `hipGetDeviceProperties`
- `hipEventCreate`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipEventElapsedTime`
- `hipEventDestroy`

## Data Types and Enums

- `rocwmma::float64_t`
- `rocwmma::row_major`
- `rocwmma::col_major`
