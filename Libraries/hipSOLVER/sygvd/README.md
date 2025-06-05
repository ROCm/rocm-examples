# hipSOLVER Generalized Symmetric Eigenvalue Problem Solver Example

## Description

This example illustrates how to solve the generalized symmetric-definite eigenvalue problem for a given pair of matrices $(A,B)$ using hipSOLVER.

Given a pair $(A,B)$ such that

- $A,B \in \mathcal{M}_n(\mathbb{R})$ are symmetric matrices and
- $B$ is [positive definite](https://en.wikipedia.org/wiki/Definite_matrix),

the said problem consists on solving the following equation:

$(A - \lambda B)x = 0$.

Such a solution is given by a pair $(X, \Lambda)$, where

- $X$ is an $n \times n$ orthogonal matrix containing (as columns) the eigenvectors $x_i$ for $i = 0, \dots, n-1$ and
- $\Lambda$ is an $n \times n$ diagonal matrix containing the eigenvalues $\lambda_i$ for $i = 0, \dots, n-1$

such that

$(A - \lambda_i B)x_i = 0$  for $i = 0, \dots, n-1$.

### Application flow

1. Declare and initialize a number of constants for the input matrices.
2. Allocate and initialize the host matrices.
3. Allocate device memory and copy input data from host to device.
4. Create a hipSOLVER handle and query the working space size.
5. Invoke `hipsolverDsygvd` to calculate the eigenvalues and eigenvectors of the pair $(A,B)$.
6. Copy the resulting `devInfo` value to the host and check if the operation was successful.
7. Copy the resulting eigenvalues and eigenvectors to the host and print the eigenvalues.
8. Validate the solution by checking if $B \cdot X \cdot \Lambda - A \cdot X$ is the zero matrix using the hipBLAS API.
9. Free device memory and the handles.
10. Print validation result.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER is initialized by calling `hipsolverCreate(hipsolverHandle_t*)` and is terminated by calling `hipsolverDestroy(hipsolverHandle_t)`.
- `hipsolver[SD]sygvd` computes the eigenvalues and optionally the eigenvectors of an $n \times n$ symmetric pair $(A, B)$, where $B$ is also positive definite. The correct function signature should be chosen based on the datatype of the input pair:

  - `S` (single-precision real: `float`)
  - `D` (double-precision real: `double`).

  A complex version of this function is also available under the name `hipsolver[CZ]hegvd`. It accepts the same parameters as `hipsolver[SD]sygvd`, except that the correct function signature should be chosen based on the following data types:

  - `C` (single-precision complex: `hipFloatComplex`).
  - `Z` (double-precision complex: `hipDoubleComplex`).

  In this example, a double-precision real input pair is used, in which case the function accepts the following parameters:

  - `hipsolverHandle_t handle`
  - `hipsolverEigType_t itype`: Specifies the problem type to be solved:

    - `HIPSOLVER_EIG_TYPE_1`: $A \cdot X = B \cdot X \cdot \Lambda$
    - `HIPSOLVER_EIG_TYPE_2`: $A \cdot B \cdot X = X \cdot \Lambda$
    - `HIPSOLVER_EIG_TYPE_3`: $B \cdot A \cdot X = X \cdot \Lambda$

  - `hipsolverEigMode_t jobz`: Specifies whether the eigenvectors should also be calculated besides the eigenvalues. The following values are accepted:

    - `HIPSOLVER_EIG_MODE_NOVECTOR`: calculate the eigenvalues only.
    - `HIPSOLVER_EIG_MODE_VECTOR`: calculate both the eigenvalues and the eigenvectors. The eigenvectors are calculated using a divide-and-conquer algorithm and are overwritten to the device memory location pointed by `*A`.

  - `hipSolverFillMode_t uplo`: Specifies which part of input matrices $A$ and $B$ are stored. The following values are accepted:

    - `HIPSOLVER_FILL_MODE_UPPER`: The provided `*A` and `*B` pointers point to the upper triangle matrix data.
    - `HIPSOLVER_FILL_MODE_LOWER`: The provided `*A` and `*B` pointers point to the lower triangle matrix data.
  - `int n`: Dimension of matrices $A$ and $B$.
  - `double *A`: Pointer to matrix $A$ in device memory.
  - `int lda`: Leading dimension of matrix $A$.
  - `double *B`: Pointer to matrix $B$ in device memory.
  - `int ldb`: Leading dimension of matrix $B$.
  - `double *W`: Pointer to vector in device memory representing the diagonal of matrix $\Lambda$, where the resulting eigenvalues are written.
  - `double *work`: Pointer to working space.
  - `int lwork`: Size of the working space, obtained with `hipsolverDsygvd_bufferSize`.
  - `int *devInfo`: Convergence result of the function. If 0, the algorithm converged. If greater than 0 and:

    - `devInfo = i` for $0 < i \leq n$, then `devInfo` elements of the intermediate tridiagonal matrix did not converge to 0.
    - `devInfo = n + i` for $0 < i \leq n$, the leading minor of order $i$ of $B$ is not positive definite.

  Return type: `hipsolverStatus_t`.

- `hipsolver[SD]sygvd_bufferSize` allows to obtain the size (in bytes) needed for the working space of the `hipsolver[SD]sygvd` function. The character matched in `[SD]` coincides with the one in `hipsolver[SD]sygvd`.

  This function accepts the following input parameters:

  - `hipsolverHandle_t handle`
  - `hipsolverEigType_t itype`: Specifies the problem type to be solved.
  - `hipsolverEigMode_t jobz`: Specifies whether the eigenvectors should also be calculated besides the eigenvalues.
  - `hipSolverFillMode_t uplo`: Specifies whether the upper or lower triangles of the of the symmetric input matrices $A$ and $B$ are stored.
  - `int n`: Simension of matrices $A$ and $B$.
  - `double *A`: Pointer to matrix $A$ in device memory.
  - `int lda`: Leading dimension of matrix $A$.
  - `double *B`: Pointer to matrix $B$ in device memory.
  - `int ldb`: Leading dimension of matrix $B$.
  - `double *W`: Pointer to vector in device memory representing the diagonal of matrix $\Lambda$, where the resulting eigenvalues are written.
  - `int *lwork`: The required buffer size is written to this location.

  Return type: `hipsolverStatus_t`.

### hipBLAS

- For validating the solution we have used the hipBLAS functions `hipblasDdgmm` and `hipblasDgemm`. `hipblasDgemm` is showcased (strided-batched and with single-precision real type) in the [gemm_strided_batched example](/Libraries/hipBLAS/gemm_strided_batched/).

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

- `hipsolverCreate`
- `hipsolverDsygvd_bufferSize`
- `hipsolverDsygvd`
- `hipsolverDestroy`
- `hipsolverHandle_t`
- `HIPSOLVER_EIG_MODE_VECTOR`
- `HIPSOLVER_EIG_TYPE_1`
- `HIPSOLVER_FILL_MODE_UPPER`

### hipBLAS

- `hipblasCreate`
- `hipblasDestroy`
- `hipblasDdgmm`
- `hipblasDgemm`
- `hipblasHandle_t`
- `HIPBLAS_OP_N`
- `HIPBLAS_POINTER_MODE_HOST`
- `hipblasSetPointerMode`
- `HIPBLAS_SIDE_RIGHT`

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
