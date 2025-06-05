# hipBLAS Level 3 Generalized Matrix Multiplication Strided Batched Example

## Description

This example illustrates the use of the hipBLAS Level 3 Strided Batched General Matrix Multiplication. The hipBLAS GEMM STRIDED BATCHED performs a matrix--matrix operation for a _batch_ of matrices as:

$C[i] = \alpha \cdot op_a(A[i]) \cdot op_b(B[i]) + \beta \cdot (C[i])$

for each $i \in [0, batch - 1]$, where $X[i] = X + i \cdot strideX$ is the $i$-th element of the correspondent batch and $op_m(M)$ is one of the following:

- $op_m(M) = M$ or
- $op_m(M) = M^T$ (transpose $M$: $M_{ij}^T = M_{ji}$) or
- $op_m(M) = M^H$ (Hermitian $M$: $M_{ij}^H = \bar M_{ji} $).
In this example the identity is used.

$\alpha$ and $\beta$ are scalars, and $A$, $B$ and $C$ are the batches of matrices. For each $i$, $A[i]$, $B[i]$ and $C[i]$ are matrices such that
$op_a(A[i])$ is an $m \times k$ matrix, $op_b(B[i])$ a $k \times n$ matrix and $C_i$ an $m \times n$ matrix.

### Application flow

1. Read in command-line parameters.
2. Set dimension variables of the matrices and get the batch count.
3. Allocate and initialize the host matrices. Set up $B$ matrix as an identity matrix.
4. Initialize gold standard matrix.
5. Compute CPU reference result with strided batched subvectors.
6. Allocate device memory.
7. Copy data from host to device.
8. Create a hipBLAS handle.
9. Invoke the hipBLAS GEMM STRIDED BATCHED function.
10. Copy the result from device to host.
11. Destroy the hipBLAS handle, release device memory.
12. Validate the output by comparing it to the CPU reference result.

### Command line interface

The application provides the following optional command line arguments:

- `-a` or `--alpha`. The scalar value $\alpha$ used in the GEMM operation. Its default value is 1.
- `-b` or `--beta`. The scalar value $\beta$ used in the GEMM operation. Its default value is 1.
- `-c` or `--count`. Batch count. Its default value is 3.
- `-m` or `--m`. The number of rows of matrices $A$ and $C$, which must be greater than 0. Its default value is 5.
- `-n` or `--n`. The number of columns of matrices $B$ and $C$, which must be greater than 0. Its default value is 5.
- `-k` or `--k`. The number of columns of matrix $A$ and rows of matrix $B$, which must be greater than 0. Its default value is 5.

## Key APIs and Concepts

- The performance of a numerical multi-linear algebra code can be heavily increased by using tensor contractions [ [Y. Shi et al., HiPC, pp 193, 2016.](https://doi.org/10.1109/HiPC.2016.031) ], thereby most of the hipBLAS functions have a`_batched` and a `_strided_batched` [ [C. Jhurani and P. Mullowney, JPDP Vol 75, pp 133, 2015.](https://doi.org/10.1016/j.jpdc.2014.09.003) ] extensions.<br/>
We can apply the same multiplication operator for several matrices if we combine them into batched matrices. Batched matrix multiplication has a performance improvement for a large number of small matrices. For a constant stride between matrices, further acceleration is available by strided batched GEMM.
- hipBLAS is initialized by calling `hipblasCreate(hipblasHandle*)` and it is terminated by calling `hipblasDestroy(hipblasHandle)`.
- The _pointer mode_ controls whether scalar parameters must be allocated on the host (`HIPBLAS_POINTER_MODE_HOST`) or on the device (`HIPBLAS_POINTER_MODE_DEVICE`). It is controlled by `hipblasSetPointerMode`.
- The symbol $op(M)$ denotes the following operations, as defined in the Description section:
  - `HIPBLAS_OP_N`: identity operator ($op(M) = M$),
  - `HIPBLAS_OP_T`: transpose operator ($op(M) = M^T$) or
  - `HIPBLAS_OP_C`: Hermitian (conjugate transpose) operator ($op(M) = M^H$).
- `hipblasStride` strides between matrices or vectors in strided_batched functions.
- `hipblas[HSDCZ]gemmStridedBatched`

    Depending on the character matched in `[HSDCZ]`, the norm can be obtained with different precisions:
  - `H`(half-precision: `hipblasHalf`)
  - `S` (single-precision: `float`)
  - `D` (double-precision: `double`)
  - `C` (single-precision complex: `hipComplex`)
  - `Z` (double-precision complex: `hipDoubleComplex`).

    Input parameters for `hipblasSgemmStridedBatched`:
    - `hipblasHandle_t handle`
    - `hipblasOperation_t trans_a`: transformation operator on each $A_i$ matrix
    - `hipblasOperation_t trans_b`: transformation operator on each $B_i$ matrix
    - `int m`: number of rows in each $op_a(A[i])$ and $C$ matrices
    - `int n`: number of columns in each $op_b(B[i])$ and $C$ matrices
    - `int k`: number of columns in each $op_a(A[i])$ matrix and number of rows in each $op_b(B[i])$ matrix
    - `const float *alpha`: scalar multiplier of each $C_i$ matrix addition
    - `const float  *A`: pointer to the each $A_i$ matrix
    - `int lda`: leading dimension of each $A_i$ matrix
    - `long long stride_a`: stride size for each $A_i$ matrix
    - `const float  *B`: pointer to each $B_i$ matrix
    - `int ldb`: leading dimension of each $B_i$ matrix
    - `const float  *beta`: scalar multiplier of the $B \cdot C$ matrix product
    - `long long stride_b`: stride size for each $B_i$ matrix
    - `float  *C`: pointer to each $C_i$ matrix
    - `int ldc`: leading dimension of each $C_i$ matrix
    - `long long stride_c`: stride size for each $C_i$ matrix
    - `int batch_count`: number of matrices

  - `hipblasHandle_t handle`
  - `hipblasOperation_t trans_a`: transformation operator on each $A_i$ matrix
  - `hipblasOperation_t trans_b`: transformation operator on each $B_i$ matrix
  - `int m`: number of rows in each $A_i'$ and $C$ matrices
  - `int n`: number of columns in each $B_i'$ and $C$ matrices
  - `int k`: number of columns in each $A_i'$ matrix and number of rows in each $B_i'$ matrix
  - `const float *alpha`: scalar multiplier of each $C_i$ matrix addition
  - `const float  *A`: pointer to the each $A_i$ matrix
  - `int lda`: leading dimension of each $A_i$ matrix
  - `long long stride_a`: stride size for each $A_i$ matrix
  - `const float  *B`: pointer to each $B_i$ matrix
  - `int ldb`: leading dimension of each $B_i$ matrix
  - `const float  *beta`: scalar multiplier of the $B \cdot C$ matrix product
  - `long long stride_b`: stride size for each $B_i$ matrix
  - `float  *C`: pointer to each $C_i$ matrix
  - `int ldc`: leading dimension of each $C_i$ matrix
  - `long long stride_c`: stride size for each $C_i$ matrix
  - `int batch_count`: number of matrices

  Return value: `hipblasStatus_t`

## Demonstrated API Calls

### hipBLAS

- `hipblasCreate`
- `hipblasDestroy`
- `hipblasHandle_t`
- `hipblasSgemmStridedBatched`
- `hipblasOperation_t`
- `hipblasStride`
- `hipblasSetPointerMode`
- `HIPBLAS_OP_N`
- `HIPBLAS_POINTER_MODE_HOST`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
