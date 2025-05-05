# hipSOLVER Compatibility API Symmetric Eigenvalue Calculation (divide and conquer algorithm)

## Description

This example illustrates how to solve the standard symmetric-definite eigenvalue problem for a symmetric matrix $A$ using hipSOLVER's [Compatibility API](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/reference/compat-api/lapacklike.html). This API offers wrapper functions for the ones existing in hipSOLVER (and their equivalents in [cuSolverDN](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-dense-lapack)) and is intended to be used when porting cuSOLVER applications to hipSOLVER ones. The main advantage of this API is that its functions follow the same method signature format as cuSolverDN's, which makes easier the port.

Given an $n \times n$ symmetric matrix $A$, the said problem consists on solving the following equation:

$Ax = \lambda x$.

A solution for this problem is given by a pair $(X, \Lambda)$, where

- $X$ is an $n \times n$ orthogonal matrix containing (as columns) the eigenvectors $x_i$ for $i = 0, \dots, n-1$ and
- $\Lambda$ is an $n \times n$ diagonal matrix containing the eigenvalues $\lambda_i$ for $i = 0, \dots, n-1$

such that

$A x_i = \lambda_i x_i$  for $i = 0, \dots, n-1$.

### Application flow

1. Declare and initialize a number of constants for the input matrix.
2. Allocate and initialize the host matrix $A$.
3. Allocate device memory and copy input data from host to device.
4. Create a hipSOLVER compatibility API handle and query the working space size.
5. Invoke `hipsolverDnDsyevdx` to calculate the eigenvalues and eigenvectors of matrix $A$.
6. Copy the resulting devInfo value `d_syevdx_info` value to the host and check if the operation was successful.
7. Copy the resulting eigenvalues and eigenvectors to the host and print the eigenvalues.
8. Validate the solution by checking if $X \cdot \Lambda - A \cdot X$ is the zero matrix using the hipBLAS API.
9. Free device memory and the handles.
10. Print validation result.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER is initialized by calling `hipsolverDnCreate(hipsolverHandle_t*)` and it is terminated by calling `hipsolverDnDestroy(hipsolverHandle_t)`.
- In this example `hipsolverDnHandle_t` is used instead of `hipsolverHandle_t`. `hipsolverDnHandle_t` is actually a typedef of `hipsolverHandle_t`, so they can be used equivalently.
- `hipsolverDn[SD]syevdx` computes the eigenvalues of an $n \times n$ symmetric matrix $A$. The correct function signature should be chosen based on the datatype of the input matrix:

  - `S` (single-precision real: `float`)
  - `D` (double-precision real: `double`)

  For single- and double-precision complex values, the function `hipsolverDn[CZ]heevdx(...)` is available in hipSOLVER.

  In this example, a double-precision real input matrix is used, in which case the function accepts the following parameters:

  - `hipsolverHandle_t handle`
  - `hipsolverEigMode_t jobz`: Specifies whether the eigenvectors should also be calculated besides the eigenvalues. The following values are accepted:

    - `HIPSOLVER_EIG_MODE_NOVECTOR`: Calculate the eigenvalues only.
    - `HIPSOLVER_EIG_MODE_VECTOR`: Calculate both the eigenvalues and the eigenvectors. The eigenvectors are calculated by a divide and conquer algorithm and are written to the memory location specified by `*A`.

  - `hipsolverEigRange_t range`: Specifies a range of eigenvalues to be returned. The following values are accepted:

    - `HIPSOLVER_EIG_RANGE_ALL`: The whole spectrum is returned.
    - `HIPSOLVER_EIG_RANGE_V`: Only the eigenvalues in the interval `(vl, vu]` are returned. `vl` $>$ `vu` must be satisfied.
    - `HIPSOLVER_EIG_RANGE_I`: Only the eigenvalues from the `il`-th to the `iu`-th are returned. $1$ $\leq$ `il` $\leq$ `iu` $\leq$ $n$ must be satisfied.

    - `hipSolverFillMode_t uplo`: Specifies whether the upper or lower triangle of the symmetric matrix is stored. The following values are accepted:

    - `HIPSOLVER_FILL_MODE_UPPER`: The provided `*A` pointer points to the upper triangle matrix data.
    - `HIPSOLVER_FILL_MODE_LOWER`: The provided `*A` pointer points to the lower triangle matrix data.

  - `int n`: Number of rows and columns of $A$.
  - `double *A`: Pointer to matrix $A$ in device memory.
  - `int lda`: Leading dimension of matrix $A$.
  - `double vl`: Lower bound of the interval to be searched for eigenvalues if `range` = `HIPSOLVER_EIG_RANGE_V`.
  - `double vu`: Upper bound of the interval to be searched for eigenvalues if `range` = `HIPSOLVER_EIG_RANGE_V`.
  - `int il`: Smallest index of the eigenvalues to be returned if `range` = `HIPSOLVER_EIG_RANGE_I`.
  - `int iu`: Largest index of the eigenvalues to be returned if `range` = `HIPSOLVER_EIG_RANGE_I`.
  - `int *nev`: Number of eigenvalues returned.
  - `double *W`: Pointer to array $W$ in device memory, where the resulting eigenvalues are written.
  - `double *work`: Pointer to working space in device memory.
  - `int lwork`: Size of the working space.
  - `int *devInfo`: Convergence result of the function in device memory.

    - If 0, the algorithm converged.
    - If greater than 0 (`devInfo = i` for $0 < i \leq n$), then `devInfo` eigenvectors did not converge.
    - For CUDA backend, if lesser than 0 (`devInfo = -i` for $0 < i \leq n$) then the the $i^{th}$ parameter is wrong (not counting the handle).

  Return type: `hipsolverStatus_t`.

- `hipsolverDn[SD]syevdx` internally calls to `cusolverDn[SD]syevdx` for CUDA backend and to a rocSOLVER's internal `syevx` function (not the one from the public API) for ROCm backend, as no `hipsolver[SD]syevdx` function exists yet in hipSOLVER regular API.
- `hipsolverDn[SD]syevdx_bufferSize` allows to obtain the size (in bytes) needed for the working space for the `hipsolverDn[SD]syevdx` function. The character matched in `[SD]` coincides with the one in `hipsolverDn[SD]syevdx`.

  This function accepts the following input parameters:

  - `hipsolverHandle_t handle`
  - `hipsolverEigMode_t jobz`: Specifies whether the eigenvectors should also be calculated besides the eigenvalues.
  - `hipsolverEigRange_t range`: Specifies a range of eigenvalues to be returned.
  - `hipSolverFillMode_t uplo`: Specifies whether the upper or lower triangle of the symmetric matrix is stored.
  - `int n`: Number of rows and columns of $A$.
  - `double *A`: Pointer to matrix $A$ in device memory.
  - `int lda`: Leading dimension of matrix $A$.
  - `double vl`: Lower bound of the interval to be searched for eigenvalues if `range` = `HIPSOLVER_EIG_RANGE_V`.
  - `double vu`: Upper bound of the interval to be searched for eigenvalues if `range` = `HIPSOLVER_EIG_RANGE_V`.
  - `int il`: Smallest index of the eigenvalues to be returned if `range` = `HIPSOLVER_EIG_RANGE_I`.
  - `int iu`: Largest index of the eigenvalues to be returned if `range` = `HIPSOLVER_EIG_RANGE_I`.
  - `int *nev`: Number of eigenvalues returned.
  - `double *W`: Pointer to array $W$ in device memory, where the resulting eigenvalues are written.
  - `int *lwork`: The required buffer size is written to this location.

  Return type: `hipsolverStatus_t`.

### hipBLAS

- For validating the solution we have used the hipBLAS functions `hipblasDdgmm` and `hipblasDgemm`. `hipblasDgemm` computes a general scaled matrix multiplication $\left(C = \alpha \cdot A \cdot B + \beta \cdot C\right)$ and is showcased (strided-batched and with single-precision real type) in the [gemm_strided_batched example](/Libraries/hipBLAS/gemm_strided_batched/).

  - The `hipblas[SDCZ]dgmm` function performs a matrix--matrix operation between a diagonal matrix and a general $m \times n$ matrix. The order of the multiplication can be determined using a `hipblasSideMode_t` type parameter:

    - `HIPBLAS_SIDE_RIGHT`: the operation performed is $C = A \cdot diag(x)$. This is the one used in the example for computing $X \cdot \Lambda$.
    - `HIPBLAS_SIDE_LEFT`: the operation performed is $C = diag(x) \cdot A$.

    The correct function signature should be chosen based on the datatype of the input matrices:

    - `S` (single-precision real: `float`)
    - `D` (double-precision real: `double`)
    - `C` (single-precision complex: `hipComplex`)
    - `Z` (double-precision complex: `hipDoubleComplex`).

    Return type: `hipblasStatus_t`.

- The `hipblasPointerMode_t` type controls whether scalar parameters must be allocated on the host (`HIPBLAS_POINTER_MODE_HOST`) or on the device (`HIPBLAS_POINTER_MODE_DEVICE`). It is set by using `hipblasSetPointerMode`.

## Used API surface

### hipSOLVER

- `HIPSOLVER_EIG_MODE_VECTOR`
- `HIPSOLVER_FILL_MODE_UPPER`

### hipSOLVER Compatibility API

- `HIPSOLVER_EIG_RANGE_I`
- `hipsolverDnCreate`
- `hipsolverDnDestroy`
- `hipsolverDnDsyevdx`
- `hipsolverDnDsyevdx_bufferSize`
- `hipsolverDnHandle_t`

### hipBLAS

- `HIPBLAS_OP_N`
- `HIPBLAS_POINTER_MODE_HOST`
- `HIPBLAS_SIDE_RIGHT`
- `hipblasCreate`
- `hipblasDdgmm`
- `hipblasDestroy`
- `hipblasDgemm`
- `hipblasHandle_t`
- `hipblasSetPointerMode`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
