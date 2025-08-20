# rocWMMA Simple Single-Precision General Matrix Multiplication (SGEMM)

## Description

This example demonstrates a basic single-precision General Matrix Multiplication (GEMM) using rocWMMA, showcasing the library's support for FP32 computations on AMD GPUs.

The operation calculates the following product:

$D = \alpha \cdot A \cdot B + \beta \cdot C$

where:

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$

## Application flow

1. **Device Capability Check**: The application checks if the device supports single-precision (FP32) operations.
2. **Matrix Initialization**: Host-side matrices (A, B, C, D) are allocated using `std::vector<float32_t>` and initialized with random data.
3. **Device Memory Management**: Device memory is allocated for all matrices, and the input matrices are copied from host to device.
4. **Kernel Launch**: The `sgemm_rocwmma_d` kernel is launched with a 2D grid of thread blocks. Each warp within a block is responsible for computing a tile of the output matrix.
5. **GEMM Computation**: Inside the kernel, each warp:
    - Initializes an accumulator fragment to zero.
    - Loops over the K dimension of the input matrices.
    - In each iteration, loads tiles of A and B into rocWMMA fragments using `rocwmma::load_matrix_sync`.
    - Performs the matrix multiplication and accumulation on the fragments using `rocwmma::mma_sync`.
    - After the loop, loads the corresponding tile of matrix C.
    - Applies the alpha and beta scaling to the accumulated result and C.
    - Stores the final result back to global memory using `rocwmma::store_matrix_sync`.
6. **Result Verification**: The output matrix D is copied back to the host and compared against a CPU-based reference implementation.
7. **Cleanup**: All device memory is freed.

## Key APIs and Concepts

- **Single-Precision (FP32) Computation**: This example uses `float32_t` for all matrix data and computations. FP32 offers a good balance between performance and precision and is widely supported on GPUs.

- **rocWMMA Fragments**:
  - `rocwmma::fragment`: Represents a tile of a matrix held in a warp's registers. The size of the fragment (e.g., 16x16) is a key performance parameter.
  - Fragments are defined for matrices A, B, and the accumulator, all with `float32_t` data type.

- **Core rocWMMA Operations**:
  - `rocwmma::load_matrix_sync()`: Loads a tile of a matrix from global memory into a fragment.
  - `rocwmma::mma_sync()`: Performs the matrix multiply-accumulate operation on the fragments.
  - `rocwmma::store_matrix_sync()`: Stores the result from an accumulator fragment back to global memory.
  - `rocwmma::fill_fragment()`: Initializes an accumulator fragment with a value.

- **Kernel Tiling Strategy**: The overall matrix multiplication is decomposed into smaller tile-based computations. Each warp computes one tile of the output matrix by iterating through tiles of the input matrices along the K dimension.

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

- `rocwmma::float32_t`
- `rocwmma::row_major`
- `rocwmma::col_major`
