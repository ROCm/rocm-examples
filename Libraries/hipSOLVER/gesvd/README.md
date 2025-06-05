# hipSOLVER Singular Value Decomposition Example

## Description

This example illustrates the use of the hipSOLVER Singular Value Decomposition functionality. The hipSOLVER `gesvd` computes the singular values and optionally the left and right singular vectors of an $m \times n$ matrix $A$. The [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) is then given by $A = U \cdot S \cdot V^H$, where:

- $U$ is an $m \times m$ orthonormal matrix. Its column vectors are known as _left singular vectors_ of $A$ and correspond to the eigenvectors of the Hermitian and positive semi-definite matrix $AA^H$.

- $S$ is an $m \times n$ diagonal matrix with non-negative real numbers on the diagonal, the _singular values_ of $A$, defined as the (positive) square roots of the eigenvalues of the Hermitian and positive semi-definite matrix $A^HA$. Note that we always have $rank(A)$ non-zero singular values.

- $V^H$ is the Hermitian transpose of an $n \times n$ orthonormal matrix, $V$. Its row vectors are known as the _right singular vectors_ of $A$ and are defined as the eigenvectors of the Hermitian and positive semi-definite matrix $A^HA$.

### Application flow

1. Parse command line arguments for the dimension of the input matrix.
2. Declare and initialize a number of constants for the input and output matrices and vectors.
3. Allocate and initialize the host matrices and vectors.
4. Allocate device memory and copy input data from host to device.
5. Define how we want to obtain the singular vectors. In this example matrices $U$ and $V^H$ are written in their entirety.
6. Create a hipSOLVER handle and query the working space size.
7. Invoke the hipSOLVER `gesvd` function with double precision.
8. Copy the results from device to host.
9. Print trace message for convergence of the BDSQR function.
10. Validate the solution by checking if $U \cdot S \cdot V^H - A$ is the zero matrix using the hipBLAS API.
11. Free device memory and the handles.

### Command line interface

The application provides the following optional command line arguments:

- `--n <n>`. Number of rows of input matrix $A$, the default value is `3`.
- `--m <m>`. Number of columns of input matrix $A$, the default value is `2`.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER is initialized by calling `hipsolverCreate(hipsolverHandle_t*)` and it is terminated by calling `hipsolverDestroy(hipsolverHandle_t)`.

- `hipsolver[SDCZ]gesvd` computes the singular values and optionally the left and right singular vectors of an $m \times n$ matrix $A$. The correct function signature should be chosen based on the datatype of the input matrix:

  - `S` (single-precision real: `float`)
  - `D` (double-precision real: `double`)
  - `C` (single-precision complex: `hipFloatComplex`)
  - `Z` (double-precision complex: `hipDoubleComplex`).

  In this example, a double-precision real input matrix is used, in which case the function accepts the following parameters:

  - `hipsolverHandle_t handle`
  - `signed char jobu` and `signed char jobv`: define how the left and right singular vectors, respectively, are calculated and stored. The following values are accepted:

    - `'A'`: all columns of $U$, or rows of $V^H$, are calculated.
    - `'S'`: only the singular vectors associated to the singular values of $A$ are calculated and stored as columns of $U$ or rows of $V^H$. In this case some columns of $U$ or rows of $V^H$ may be left unmodified.
    - `'O'`: same as `'S'`, but the singular vectors are stored in matrix $A$, overwriting it.
    - `'N'`: singular vectors are not computed.

  - `int m`: number of rows of $A$
  - `int n`: number of columns of $A$
  - `double *A`: pointer to matrix $A$
  - `int lda`: leading dimension of matrix $A$
  - `double *S`: pointer to vector $S$
  - `double *U`: pointer to matrix $U$
  - `int ldu`: leading dimension of matrix $U$
  - `double *V`: pointer to matrix $V^H$
  - `int ldv`: leading dimension of matrix $V^H$
  - `double *work`: pointer to working space.
  - `int lwork`: size of the working space.
  - `double *rwork`: unconverged superdiagonal elements of the upper bidiagonal matrix used internally for the BDSQR algorithm.
  - `int *devInfo`: convergence result of the BDSQR function. If 0, the algorithm converged, if greater than 0 then `info` elements of vector $E$ did not converge to 0.

  Return type: `hipsolverStatus_t`.

- `hipsolver[SDCZ]gesvd_bufferSize` allows to obtain the size (in bytes) needed for the working space for the `hipsolver[SDCZ]gesvd` function. The character matched in `[SDCZ]` coincides with the one in `hipsolver[SDCZ]gesvd`.

  This function accepts the following input parameters:

  - `hipsolverHandle_t handle`
  - `signed char jobu`: defines how left singular vectors are calculated and stored
  - `signed char jobv`: defines how right singular vectors are calculated and stored
  - `int m`: number of rows of $A$
  - `int n`: number of columns of $A$
  - `int lwork`: size (to be computed) of the working space.

  Return type: `hipsolverStatus_t`.

### hipBLAS

- For validating the solution we have used the hipBLAS functions `hipblasDdgmm` and `hipblasDgemm`. `hipblasDgemm` is showcased (strided-batched and with single-precision) in the [gemm_strided_batched example](/Libraries/hipBLAS/gemm_strided_batched/).

  - The `hipblas[SDCZ]dgmm` function performs a matrix--matrix operation between a diagonal matrix and a general $m \times n$ matrix. The order of the multiplication can be determined using a `hipblasSideMode_t` type parameter:

    - `HIPBLAS_SIDE_RIGHT`: the operation performed is $C = A \cdot diag(x)$.
    - `HIPBLAS_SIDE_LEFT`: the operation performed is $C = diag(x) \cdot A$. This is the one used in the example for computing $S \cdot V^H$.

    The correct function signature should be chosen based on the datatype of the input matrices:

    - `S` (single-precision real: `float`)
    - `D` (double-precision real: `double`)
    - `C` (single-precision complex: `hipComplex`)
    - `Z` (double-precision complex: `hipDoubleComplex`).

    Return type: `hipblasStatus_t`.

- The `hipblasPointerMode_t` type controls whether scalar parameters must be allocated on the host (`HIPBLAS_POINTER_MODE_HOST`) or on the device (`HIPBLAS_POINTER_MODE_DEVICE`). It is set by using `hipblasSetPointerMode`.

## Used API surface

### hipSOLVER

- `hipsolverDgesvd`
- `hipsolverDgesvd_bufferSize`
- `hipsolverHandle_t`
- `hipsolverCreate`
- `hipsolverDestroy`

### hipBLAS

- `hipblasCreate`
- `hipblasDestroy`
- `hipblasDdgmm`
- `hipblasDgemm`
- `hipblasHandle_t`
- `HIPBLAS_OP_N`
- `HIPBLAS_POINTER_MODE_HOST`
- `hipblasSetPointerMode`
- `HIPBLAS_SIDE_LEFT`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
