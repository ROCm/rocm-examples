# hipSOLVER LU Factorization Example

## Description

This example illustrates the use of the hipSOLVER LU factorization functionality. The hipSOLVER `getrf` computes the [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition) of an $m \times n$ matrix $A$, with partial pivoting. This factorization is given by $P \cdot A = L \cdot U$, where:

- `getrf()`: This is the blocked Level-3-BLAS version of the LU factorization algorithm. An optimized internal implementation without rocBLAS calls could be executed with mid-size matrices.

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
6. Create a `hipsolverHandle_t` handle.
7. Invoke the hipSOLVER `getrf` function with double precision.
8. Copy the results from device to host.
9. Print trace messages.
10. Free device memory and the hipSOLVER handle.

## Key APIs and Concepts

### hipSOLVER

- `hipsolver[SDCZ]getrf` computes the LU factorization of an $m \times n$ input matrix $A$. The correct function signature should be chosen, based on the datatype of the input matrix:

  - `S` (single-precision: `float`)
  - `D` (double-precision: `double`)
  - `C` (single-precision complex: `hipFloatComplex`)
  - `Z` (double-precision complex: `hipDoubleComplex`).

  Input parameters for the precision used in this example (double-precision):

  - `hipsolverHandle_t handle`
  - `const int m`: number of rows of $A$
  - `const int n`: number of columns of $A$
  - `double *A`: pointer to matrix $A$
  - `const int lda`: leading dimension of matrix $A$
  - `double *Ipiv`: pointer to vector $Ipiv$
  - `int *info`: result of the LU factorization. If 0, the factorization succeeded, if greater than 0 then $U$ is singular and $U[info,info]$ is the first zero pivot.

  Return type: `hipsolverStatus_t`

- `hipsolver[SDCZ]getrf_bufferSize` allows to obtain the size (in bytes) needed for the working space for the `hipsolver[SDCZ]getrf` function. The character matched in `[SDCZ]` coincides with the one in `hipsolver[SDCZ]getrf`.

  This function accepts the following input parameters:

  - `hipsolverHandle_t handle`
  - `int m` number of rows of $A$
  - `int n` number of columns of $A$
  - `double *A` pointer to matrix $A$
  - `int lda` leading dimension of matrix $A$
  - `int *lwork` returns the size of the working space required

  The return type is `hipsolverStatus_t`.

## Used API surface

### hipSOLVER

- `hipsolverHandle_t`
- `hipsolverCreate`
- `hipsolverDestroy`
- `hipsolverDgetrf_bufferSize`
- `hipsolverDgetrf`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
