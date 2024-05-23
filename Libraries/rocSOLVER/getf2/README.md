# rocSOLVER LU Factorization Example

## Description

This example illustrates the use of the rocSOLVER `getf2` functionality. The rocSOLVER `getf2` computes the [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition) of an $m \times n$ matrix $A$, with partial pivoting. This factorization is given by $P \cdot A = L \cdot U$, where:

- `getf2()`: This is the unblocked Level-2-BLAS version of the LU factorization algorithm. An optimized internal implementation without rocBLAS calls could be executed with small and mid-size matrices.
- $A$ is the $m \times n$ input matrix.
- $P$ is an $m \times m$ [permutation matrix](https://en.wikipedia.org/wiki/Permutation_matrix), in this example stored as an array of row indices `vector<int> Ipiv` of size `min(m, n)`.
- $L$ is:
  - an $m \times m$ lower triangular matrix, when $m \leq n$.
  - an $m \times n$ lower trapezoidal matrix, when $` m > n `$.
- $U$ is:
  - an $m \times n$ upper trapezoidal matrix, when $` m < n `$.
  - an $n \times n$ upper tridiagonal matrix, when $m \geq n$

### Application flow

1. Parse command line arguments for the dimension of the input matrix.
2. Declare and initialize a number of constants for the input and output matrices and vectors.
3. Allocate and initialize the host matrices and vectors.
4. Allocate device memory.
5. Copy input data from host to device.
6. Create a rocBLAS handle.
7. Invoke the rocSOLVER `getf2` function with double precision.
8. Copy the results from device to host.
9. Print trace messages.
10. Free device memory and the rocBLAS handle.

## Key APIs and Concepts

### rocSOLVER

- `rocsolver_[sdcz]getf2` computes the LU factorization of the $m \times n$ input matrix $A$. The correct function signature should be chosen, based on the datatype of the input matrix:
  - `s` (single-precision: `float`)
  - `d` (double-precision: `double`)
  - `c` (single-precision complex: `rocblas_float_complex`)
  - `z` (double-precision complex: `rocblas_double_complex`).

Input parameters for the precision used in this example (double-precision):

- `rocblas_handle handle`
- `const rocblas_int m`: number of rows of $A$
- `const rocblas_int n`: number of columns of $A$
- `rocblas_double *A`: pointer to matrix $A$
- `const rocblas_int lda`: leading dimension of matrix $A$
- `rocblas_double *Ipiv`: pointer to vector $Ipiv$
- `rocblas_int *info`: result of the getf2 function. If 0, the factorization succeeded, if greater than 0 then $U$ is singular and $U[info,info]$ is the first zero pivot.

Return type: `rocblas_status`

## Used API surface

### rocSOLVER

- `rocsolver_dgetf2`

### rocBLAS

- `rocblas_create_handle`
- `rocblas_destroy_handle`
- `rocblas_double`
- `rocblas_handle`
- `rocblas_int`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
