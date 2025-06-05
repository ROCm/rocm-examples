# hipSOLVER Symmetric Eigenvalue via Generalized Jacobi Batched Example

## Description

This example illustrates how to solve the standard symmetric-definite eigenvalue problem for a batch $A$ of $m$ symmetric matrices $A_i$ using hipSOLVER. That is, showcases how to compute the eigenvalues and eigenvectors of a batch of symmetric matrices. The eigenvectors are computed using the Jacobi method.

Given a batch of $m$ symmetric matrices $A_i$ of dimension $n$, the said problem consists on solving the following equation:

$A_ix = \lambda x$

for each $0 \leq i \leq m-1$.

A solution for this problem is given by $m$ pairs $(X_i, \Lambda_i)$, where

- $X_i$ is an $n \times n$ orthonormal matrix containing (as columns) the eigenvectors $x_{i_j}$ for $j = 0, \dots, n-1$ and
- $\Lambda_i$ is an $n \times n$ diagonal matrix containing the eigenvalues $\lambda_{i_j}$ for $j = 0, \dots, n-1$

such that

$A_i x_{i_j} = \lambda_{i_j} x_{i_j}$

for $j = 0, \dots, n-1$ and $i = 0, \dots, m-1$.

The results are verified by checking with hipBLAS that the following equality is satisfied

$A_i X_i - X_i \Lambda_i = 0$

for each $0 \leq i \leq m - 1$.

### Command line interface

The application provides the following command line arguments:

- `-h` displays information about the available parameters and their default values.
- `-n, --n <n>` sets the size of the $n \times n$ input matrices in the batch. The default value is `3`.
- `-b, --batch_count <batch_count>` sets `batch_count` as the number of matrices in the batch. The default value is `2`.

## Application flow

1. Parse command line arguments.
2. Allocate and initialize the host side inputs.
3. Allocate device memory and copy input data from host.
4. Initialize hipSOLVER by creating a handle.
5. Set parameters for hipSOLVER's `syevjBatched` function.
6. Query and allocate working space.
7. Invoke `hipsolverDsyevjBatched` to compute the eigenvalues and eigenvectors of the matrices in the batch.
8. Check returned devInfo `info` value.
9. Copy results back to host.
10. Print eigenvalues and validate solution using hipBLAS.
11. Clean up device allocations and print validation result.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER is initialized by calling `hipsolverCreate(hipsolverHandle_t*)` and it is terminated by calling `hipsolverDestroy(hipsolverHandle_t)`.
- `hipsolverEigMode_t`: specifies whether only the eigenvalues or also the eigenvectors should be computed. The following values are accepted:

  - `HIPSOLVER_EIG_MODE_VECTOR`: compute the eigenvalues and eigenvectors.
  - `HIPSOLVER_EIG_MODE_NOVECTOR`: only compute the eigenvalues.

- `hipsolverFillMode_t`: specifies whether the upper or lower triangle of each symmetric matrix is stored. The following values are accepted:

  - `HIPSOLVER_FILL_MODE_LOWER`: data is stored in the lower triangle of the matrix in the batch.
  - `HIPSOLVER_FILL_MODE_UPPER`: data is stored in the upper triangle of the matrix in the batch.

- `hipsolverCreateSyevjInfo`: initializes a structure for the input parameters and output information for calling `syevjBatched`.
- `hipsolverDestroySyevjInfo`: destroys the structure for the input parameters and output information for calling `syevjBatched`.
- `hipsolverXsyevjSetMaxSweeps`: configures the maximum amounts of sweeps.
- `hipsolverXsyevjSetTolerance`: configures  the tolerance of `syevj`.
- `hipsolverXsyevjSetSortEig` : configures whether to sort the eigenvalues (and eigenvectors, if applicable) or not. By default they are always sorted increasingly.
- `hipsolver[SD]syevjBatched_bufferSize`: computes the required workspace size `lwork` on the device for a given configuration.
- `hipsolver[SD]syevjBatched`: computes the eigenvalues of a batch $A$ of $n \times n$ symmetric matrices $A_i$. The correct function signature should be chosen based on the datatype of the input matrices:

  - `S` single-precision real (`float`)
  - `D` double-precision real (`double`)

  For single- and double-precision complex values, the function `hipsolver[CZ]heevjBatched(...)` is available in hipSOLVER.

  In this example, a double-precision real input matrix is used, in which case the function accepts the following parameters:

  - `hipsolverHandle_t handle`
  - `hipsolverEigMode_t jobz`
  - `hipsolverFillMode_t uplo`
  - `int n`: number of rows and columns of each $A_i$.
  - `double* A`: pointer to the input batch $A$ on device memory.
  - `int lda`: leading dimension of the matrices $A_i$.
  - `double* W`: pointer to array $W$ in device memory, where the resulting eigenvalues are written.
  - `double* work`: pointer to working space in device memory.
  - `int lwork`: size of the working space.
  - `int* devInfo`: pointer to where the convergence result of the function is written to in device memory.

    - If 0, the algorithm converged.
    - If greater than 0 (`devInfo = i` for $1 \leq i \leq n$), then `devInfo` eigenvectors did not converge.
    - For CUDA backend, if lesser than 0 (`devInfo = -i` for $1 \leq i \leq n$) then the the $i^{th}$ parameter is wrong (not counting the handle).

  - `syevjInfo_t params`: the structure for the input parameters and output information of `syevjBatched`.

- `hipsolverXsyevjGetSweeps`: gets the amount of executed sweeps of `syevjBatched`. Currently it's not supported for the batched version and a `HIPSOLVER_STATUS_NOT_SUPPORTED` error is emitted if this function is invoked.
- `hipsolverXsyevjGetResidual`: gets the residual of `syevjBatched`. Currently it's not supported for the batched version and a `HIPSOLVER_STATUS_NOT_SUPPORTED` error is emitted if this function is invoked.

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
- `HIPSOLVER_FILL_MODE_LOWER`
- `hipsolverCreate`
- `hipsolverCreateSyevjInfo`
- `hipsolverDestroy`
- `hipsolverDestroySyevjInfo`
- `hipsolverDsyevjBatched`
- `hipsolverDsyevjBatched_bufferSize`
- `hipsolverEigMode_t`
- `hipsolverFillMode_t`
- `hipsolverHandle_t`
- `hipsolverSyevjInfo_t`
- `hipsolverXsyevjSetMaxSweeps`
- `hipsolverXsyevjSetSortEig`
- `hipsolverXsyevjSetTolerance`

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
