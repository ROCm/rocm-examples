# hipSOLVER Generalized Dense Symmetric Eigenvalue calculation (Jacobi algorithm)

## Description

_[This example currently works only on rocSOLVER backend.](https://github.com/ROCm/hipSOLVER/issues/152)_

This example illustrates how to calculate the [generalized eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of a pair of [symmetric](https://en.wikipedia.org/wiki/Symmetric_matrix) 3x3 matrices using hipSOLVER.

The generalized eigenvalues and eigenvectors of a matrix pair are defined as:

$Av_i = \lambda_i Bv_i$

where

- $A,B\in\mathbb{R}^{n\times n}$ are symmetric matrices,
- $\lambda_i$ for $i\in\{1, \dots n\}$ are eigenvalues,
- $v_i\in\mathbb{R}^n$ are eigenvectors corresponding to the $i$-th eigenvalue (they can be normalized to unit length).

This choice corresponds to `HIPSOLVER_EIG_TYPE_1` parameter value of the solver function `hipsolverDsygvj`. Two other possibilities include $ABv_i = \lambda_i v_i$ for `HIPSOLVER_EIG_TYPE_2` and $BAv_i = \lambda_i v_i$ for `HIPSOLVER_EIG_TYPE_3`.

### Application flow

1. Instantiate two vectors of size $n\times n$ for $n=3$ containing $A$'s and $B$'s elements.
2. Allocate device memory and copy $A$'s and $B$'s elements to the device.
3. Allocate device memory for the outputs of the hipSOLVER function, namely for the calculated eigenvalue vector $W=[\lambda_1, \lambda_2, \lambda_3]$, and the returned `info` value.
4. Create a hipSOLVER handle.
5. Set configuration for the `syevj` solver, query the working space size by `hipsolverDsygvj_bufferSize` and allocate it.
6. Invoke `hipsolverDsygvj` to calculate the generalized eigenvalues of double precision symmetric matrix pair $A$ and $B$ in ascending order.
7. Copy the resulting `info` value to the host and check if the operation was successful (indicated by a `0` value).
8. Copy the resulting eigenvalues vector $W$ to the host. Print their values and check if they match the expected.
9. Copy the resulting eigenvectors matrix $V$ to the host. Print their values.
10. Copy residual and executed sweeps number to the host. Print their values.
11. Free all allocated resources.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER is initialized by calling `hipsolverCreate(hipsolverHandle_t*)` and it is terminated by calling `hipsolverDestroy(hipsolverHandle_t)`.
- `hipsolver[SD]sygvj` computes the generalized eigenvalues of an $n \times n$ matrix pair $A$ and $B$. The correct function signature should be chosen based on the datatype of the input matrix:

  - `S` (single-precision real: `float`)
  - `D` (double-precision real: `double`)

  For single- and double-precision complex values, the function `hipsolver[CZ]hegvj(...)` is available in hipSOLVER.

  In this example, a double-precision real input matrix pair is used, in which case the function accepts the following parameters:

  - `hipsolverHandle_t handle`
  - `hipsolverEigType_t itype`: Specifies the type of eigensystem problem, see [above](#description).
  - `hipsolverEigMode_t jobz`: Specifies whether the eigenvectors should also be calculated besides the eigenvalues. The following values are accepted:
    - `HIPSOLVER_EIG_MODE_NOVECTOR`: Calculate the eigenvalues only.
    - `HIPSOLVER_EIG_MODE_VECTOR`: Calculate both the eigenvalues and the eigenvectors. The eigenvectors are calculated using the Jacobi method and written to the memory location specified by `*A`.
  - `hipSolverFillMode_t uplo`: Specifies whether the upper or lower triangle of the symmetric matrix is stored. The following values are accepted:
    - `HIPSOLVER_FILL_MODE_UPPER`: The provided `*A` pointer points to the upper triangle matrix data.
    - `HIPSOLVER_FILL_MODE_LOWER`: The provided `*A` pointer points to the lower triangle matrix data.
  - `int n`: Number of rows and columns of $A$.
  - `double *A`: Pointer to matrix $A$ in device memory.
  - `int lda`: Leading dimension of matrix $A$.
  - `double *B`: Pointer to matrix $B$ in device memory.
  - `int ldb`: Leading dimension of matrix $B$.
  - `double *D`: Pointer to array $W$, where the resulting eigenvalues are written.
  - `double *work`: Pointer to working space in device memory.
  - `int lwork`: Size of the working space.
  - `int *devInfo`: Convergence result of the function in device memory. If 0, the algorithm converged, if greater than 0 and less or equal to `n` then `devInfo`-th leading minor of `B` is not positive definite, if equal to `n+1` than convergence is not achieved. Also, for CUDA backend, if `devInfo = -i` for $0 < i \leq n$, then the the $i^{th}$ parameter is wrong (not counting the handle).
  - `hipsolverSyevjInfo_t params`: Pointer to the structure with parameters of solver, that should be created by function `hipsolverCreateSyevjInfo(&params)`. Solver has two parameters:

    - Tolerance `tol`, set by function `hipsolverXsyevjSetTolerance(syevj_params, tol)`, default value of tolerance is machine zero.
    - Maximal number of sweeps to obtain convergence `max_sweeps`, set by function `hipsolverXsyevjSetMaxSweeps(syevj_params, max_sweeps)`, default value is 100.

  Return type: `hipsolverStatus_t`.

- `hipsolver[SD]sygvj_bufferSize` allows to obtain the size (in bytes) needed for the working space for the `hipsolver[SD]sygvj` function. The character matched in `[SD]` coincides with the one in `hipsolver[SD]sygvj`.

  This function accepts the following input parameters:
  - `hipsolverHandle_t handle`
  - `hipsolverEigType_t itype`: Specifies the type of eigensystem problem.
  - `hipsolverEigMode_t jobz`: Specifies whether the eigenvectors should also be calculated besides the eigenvalues.
  - `hipSolverFillMode_t uplo`: Specifies whether the upper or lower triangle of the symmetric matrix is stored.
  - `int n`: Number of rows and columns of $A$.
  - `double *A`: Pointer to matrix $A$ in device memory.
  - `int lda`: Leading dimension of matrix $A$.
  - `double *B`: Pointer to matrix $B$ in device memory.
  - `int ldb`: Leading dimension of matrix $B$.
  - `double *D`: Pointer to array $W$ in device memory, where the resulting eigenvalues are written.
  - `int *lwork`: The required buffer size is written to this location.
  - `hipsolverSyevjInfo_t params`: Pointer to the structure with parameters of solver.

  Return type: `hipsolverStatus_t`.

## Used API surface

### hipSOLVER

Types:

- `hipsolverHandle_t`
- `hipsolverSyevjInfo_t`
- `hipsolverEigType_t`
- `hipsolverEigMode_t`
- `hipsolverFillMode_t`

Functions:

- `hipsolverCreate`
- `hipsolverDestroy`
- `hipsolverCreateSyevjInfo`
- `hipsolverDestroySyevjInfo`
- `hipsolverXsyevjSetTolerance`
- `hipsolverXsyevjSetMaxSweeps`
- `hipsolverXsyevjGetResidual`
- `hipsolverXsyevjGetSweeps`
- `hipsolverDsygvj_bufferSize`
- `hipsolverDsygvj`

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
