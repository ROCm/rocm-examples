# rocWMMA Simple Single-Precision General Matrix-Vector Multiplication (SGEMV)

## Description

This example demonstrates single-precision General Matrix-Vector multiplication (GEMV) using rocWMMA. It shows how to adapt WMMA operations for matrix-vector computations with FP32 precision.

The operation calculates the following product:

$D = \alpha \cdot A \cdot B + \beta \cdot C$

where:

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a vector of dimensions $k \times 1$
- $C$ and $D$ are vectors of dimensions $m \times 1$

## Application flow

1. **Device Capability Check**: The application checks if the device supports single-precision (FP32) operations.
2. **Matrix and Vector Initialization**: A host-side matrix A and vectors B, C, and D are allocated and initialized.
3. **Device Memory Management**: Device memory is allocated for the matrix and vectors, and the inputs (A, B, C) are copied from host to device.
4. **Kernel Launch**: The `sgemv_rocwmma_d` kernel is launched. The grid and block dimensions are configured for the matrix-vector operation.
5. **GEMV Computation**: Inside the kernel, each warp:
    - Identifies the row of the output vector it is responsible for.
    - Iterates through the K dimension, loading tiles of matrix A and vector B into rocWMMA fragments.
    - Performs the matrix-vector multiplication using `rocwmma::mma_sync`.
    - Loads the corresponding element from vector C, applies the alpha and beta scaling, and stores the final result to vector D.
6. **Result Verification**: The output vector D is copied back to the host and validated against a CPU-based reference implementation.
7. **Cleanup**: Device memory is deallocated.

## Key APIs and Concepts

- **Matrix-Vector Multiplication with rocWMMA**: Although rocWMMA is designed for matrix-matrix operations, it can be adapted for matrix-vector multiplication by treating the vector as a matrix with one dimension equal to 1. In this example, the vector B is treated as a $k \times 1$ matrix.

- **Data Layout for GEMV**: Using a column-major layout for the matrix A and the vectors can be beneficial for memory access patterns in GEMV. This example uses column-major for all data.

- **rocWMMA Fragments for Vectors**: The same `rocwmma::fragment` objects are used. For the vector B, each warp repeatedly loads the same vector data as it processes different rows of matrix A.

- **Kernel Configuration**: The grid and block dimensions are adjusted for the GEMV case. The grid is essentially one-dimensional along the M dimension of the matrix, as there is only one column in the output.

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
- `rocwmma::col_major`
