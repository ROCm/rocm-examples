# rocWMMA Simple Half-Precision General Matrix Multiplication (HGEMM)

## Description

This example illustrates the use of the `rocWMMA` library to perform a basic half-precision General Matrix Multiplication (GEMM) operation. It demonstrates the fundamental usage of rocWMMA fragments and operations for mixed-precision matrix computations.

The operation calculates the following product:

$D = \alpha \cdot A \cdot B + \beta \cdot C$

where:

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$

## Application flow

1. **Matrix Initialization**: Host-side matrices are allocated and initialized. Input matrices A, B, and C use half-precision (`float16_t`).
2. **Device Memory Management**: Device memory is allocated for all matrices, and the input data is copied from host to device.
3. **Kernel Launch**: The `hgemm_rocwmma_d` kernel is launched with a 2D grid of thread blocks, where each warp is responsible for computing a tile of the output matrix D.
4. **GEMM Computation**: Inside the kernel, each warp performs the following steps:
    - Initializes an accumulator fragment to zero.
    - Loops through the K dimension of the input matrices in tile-sized steps.
    - In each iteration, it loads tiles of A and B into fragments using `rocwmma::load_matrix_sync`.
    - It performs the matrix multiplication on the fragments using `rocwmma::mma_sync`, accumulating the result in a single-precision fragment.
    - After the loop, it loads the corresponding tile of matrix C.
    - It applies the scaling factors alpha and beta to the accumulated result and C.
    - The final result is stored back to global memory using `rocwmma::store_matrix_sync`.
5. **Result Verification**: The output matrix D is copied back to the host and compared against a CPU-based reference implementation.
6. **Cleanup**: All device memory is freed.

## Key APIs and Concepts

- **Mixed Precision**: This example uses half-precision (`float16_t`) for input and output matrices (A, B, C, D) to reduce memory usage and bandwidth. The accumulation, however, is done in single-precision (`float32_t`) to maintain numerical accuracy and avoid overflow.

- **rocWMMA Fragments**:
  - `rocwmma::fragment`: Represents a tile of a matrix that is processed by a warp. The size of the fragment is defined by the template parameters (e.g., 16x16).
  - Fragments are defined for matrix A, matrix B, and the accumulator. The data types of the fragments can be different to support mixed-precision computation.

- **Core rocWMMA Operations**:
  - `rocwmma::load_matrix_sync()`: Loads a tile from global memory into a fragment's registers.
  - `rocwmma::mma_sync()`: Performs the matrix multiply-accumulate operation on the fragments held by the warp.
  - `rocwmma::store_matrix_sync()`: Stores the result from an accumulator fragment back to global memory.
  - `rocwmma::fill_fragment()`: Initializes an accumulator fragment to a specific value.

- **Tiling and Kernel Grid**: The overall matrix multiplication is broken down into smaller tiles that are computed by individual warps. The HIP kernel is launched with a 2D grid of thread blocks to cover the entire output matrix.

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

- `rocwmma::float16_t`
- `rocwmma::float32_t`
- `rocwmma::row_major`
- `rocwmma::col_major`
