# rocBLAS Level 3 Generalized Matrix Multiplication Example

## Description

This example illustrates the use of the rocBLAS Level 3 General Matrix Multiplication. The rocBLAS GEMM performs a matrix--matrix operation as:
$C = \alpha \cdot op_a(A) \cdot op_b(B) + \beta \cdot C$,
where $op_m(M)$ is one of the following:

- $op_m(M) = M$ or
- $op_m(M) = M^T$ (transpose $M$: $M_{ij}^T = M_{ji}$) or
- $op_m(M) = M^H$ (Hermitian $M$: $M_{ij}^H = \bar{M_{ji}} $),
In this example the identity is used.

$\alpha and $\beta$ are scalars, and $A$, $B$ and $C$ are matrices, with
$op_a(A)$ an $m \times k$ matrix, $op_b(B)$ a $k \times n$ matrix and $C$ an $m \times n$ matrix.

### Application flow

1. Read in command-line parameters.
2. Set dimension variables of the matrices.
3. Allocate and initialize the host matrices. Set up $B$ matrix as an identity matrix.
4. Initialize gold standard matrix.
5. Compute CPU reference result.
6. Allocate device memory.
7. Copy data from host to device.
8. Create a rocBLAS handle.
9. Invoke the rocBLAS GEMM function.
10. Copy the result from device to host.
11. Destroy the rocBLAS handle, release device memory.
12. Validate the output by comparing it to the CPU reference result.

### Command line interface

The application provides the following optional command line arguments:

- `-a` or `--alpha`. The scalar value $\alpha$ used in the GEMM operation. Its default value is 1.
- `-b` or `--beta`. The scalar value $\beta$ used in the GEMM operation. Its default value is 1.
- `-m` or `--m`. The number of rows of matrices $A$ and $C$, which must be greater than 0. Its default value is 5.
- `-n` or `--n`. The number of columns of matrices $B$ and $C$, which must be greater than 0. Its default value is 5.
- `-k` or `--k`. The number of columns of matrix $A$ and rows of matrix $B$, which must be greater than 0. Its default value is 5.

## Key APIs and Concepts

- rocBLAS is initialized by calling `rocblas_create_handle(rocblas_handle*)` and it is terminated by calling `rocblas_destroy_handle(rocblas_handle)`.

- The _pointer mode_ controls whether scalar parameters must be allocated on the host (`rocblas_pointer_mode_host`) or on the device (`rocblas_pointer_mode_device`). It is controlled by `rocblas_set_pointer_mode`.

- `rocblas_[sdhcz]gemm`

  Depending on the character matched in `[sdhcz]`, the norm can be obtained with different precisions:
  - `s` (single-precision: `rocblas_float`)
  - `d` (double-precision: `rocblas_double`)
  - `h` (half-precision: `rocblas_half`)
  - `c` (single-precision complex: `rocblas_complex`)
  - `z` (double-precision complex: `rocblas_double_complex`).

  Input parameters:

  - `rocblas_handle handle`
  - `rocblas_operation transA`: transformation operator on $A$ matrix
  - `rocblas_operation transB`: transformation operator on $B$ matrix
  - `rocblas_int m`: number of rows in $A'$ and $C$ matrices
  - `rocblas_int n`: number of columns in $B'$ and $C$ matrices
  - `rocblas_int k`: number of columns in $A'$ matrix and number of rows in $B'$ matrix
  - `const float *alpha`: scalar multiplier of $C$ matrix addition
  - `const float *A`: pointer to the $A$ matrix
  - `rocblas_int lda`: leading dimension of $A$ matrix
  - `const float *B`: pointer to the $B$ matrix
  - `rocblas_int ldb`: leading dimension of $B$ matrix
  - `const float *beta`: scalar multiplier of the $B \cdot C$ matrix product
  - `float *C`: pointer to the $C$ matrix
  - `rocblas_int ldc`: leading dimension of $C$ matrix

  Return value: `rocblas_status`

## Demonstrated API Calls

### rocBLAS

- `rocblas_int`
- `rocblas_float`
- `rocblas_operation`
- `rocblas_operation_none`
- `rocblas_handle`
- `rocblas_create_handle`
- `rocblas_destroy_handle`
- `rocblas_set_pointer_mode`
- `rocblas_pointer_mode_host`
- `rocblas_sgemm`

### HIP runtime

- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
