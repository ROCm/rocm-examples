# hipBLAS Level 3 Generalized Matrix Multiplication Strided Batched Example

## Description
This example illustrates the use of the hipBLAS Level 3 Strided Batched General Matrix Multiplication. The hipBLAS GEMM STRIDED BATCHED performs a matrix--matrix operation for a _batch_ of matrices as:

$C[i] = \alpha \cdot f(A[i]) \cdot f(B[i]) + \beta \cdot (C[i])$

for each $i \in [0, batch - 1]$, where $X[i] = X + i \cdot strideX$ is the $i$-th element of the correspondent batch and $f(X)$ is one of the following:
- $f(X) = X$ or
- $f(X) = X^T$ (transpose $X$: $X_{ij}^T = X_{ji}$) or
- $f(X) = X^H$ (Hermitian $X$: $X_{ij}^H = \bar X_{ji} $).

$\alpha$ and $\beta$ are scalars, and $A$, $B$ and $C$ are the batches of matrices. For each $i$, $A[i]$, $B[i]$ and $C[i]$ are matrices such that
$f(A[i])$ is an $m \times k$ matrix, $f(B[i])$ a $k \times n$ matrix and $C[i]$ an $m \times n$ matrix.


### Application flow
1. Read in command-line parameters.
2. Set $f$ operation, set sizes of matrices and get batch count.
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
- `-m` or `--m`. The number of rows of matrices $f(A)$ and $C$, which must be greater than 0. Its default value is 5.
- `-n` or `--n`. The number of columns of matrices $f(B)$ and $C$, which must be greater than 0. Its default value is 5.
- `-k` or `--k`. The number of columns of matrix $f(A)$ and rows of matrix $f(B)$, which must be greater than 0. Its default value is 5.

## Key APIs and Concepts
- The performance of a numerical multi-linear algebra code can be heavily increased by using tensor contractions [ [Y. Shi et al., HiPC, pp 193, 2016.](https://doi.org/10.1109/HiPC.2016.031) ], thereby most of the hipBLAS functions have a`_batched` and a `_strided_batched` [ [C. Jhurani and P. Mullowney, JPDP Vol 75, pp 133, 2015.](https://doi.org/10.1016/j.jpdc.2014.09.003) ] extensions.<br/>
We can apply the same multiplication operator for several matrices if we combine them into batched matrices. Batched matrix multiplication has a performance improvement for a large number of small matrices. For a constant stride between matrices, further acceleration is available by strided batched GEMM.
- hipBLAS is initialized by calling `hipblasCreate(hipblasHandle*)` and it is terminated by calling `hipblasDestroy(hipblasHandle)`.
- The _pointer mode_ controls whether scalar parameters must be allocated on the host (`HIPBLAS_POINTER_MODE_HOST`) or on the device (`HIPBLAS_POINTER_MODE_DEVICE`). It is controlled by `hipblasSetPointerMode`.
- The $f$ operator -- defined in Description section -- can be
    - `HIPBLAS_OP_N`: identity operator ($f(X) = X$),
    - `HIPBLAS_OP_T`: transpose operator ($f(X) = X^T$) or
    - `HIPBLAS_OP_C`: Hermitian (conjugate transpose) operator ($f(X) = X^H$).
- `hipblasStride` strides between matrices or vectors in strided_batched functions.
- `hipblas[HSDCZ]gemmStridedBatched`

    Depending on the character matched in `[HSDCZ]`, the norm can be obtained with different precisions:
    - `H`(half-precision: `hipblasHalf`)
    - `S` (single-precision: `float`)
    - `D` (double-precision: `double`)
    - `C` (single-precision complex: `hipblasComplex`)
    - `Z` (double-precision complex: `hipblasDoubleComplex`).

    Input parameters for `hipblasSgemmStridedBatched`:
    - `hipblasHandle_t handle`
    - `hipblasOperation_t trans_a`: transformation operator on each $A_i$ matrix
    - `hipblasOperation_t trans_b`: transformation operator on each $B_i$ matrix
    - `int m`: number of rows in each $f(A_i)$ and $C$ matrices
    - `int n`: number of columns in each $f(B_i)$ and $C$ matrices
    - `int k`: number of columns in each $f(A_i)$ matrix and number of rows in each $f(B_i)$ matrix
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

    Return value: `hipblasStatus_t `

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
