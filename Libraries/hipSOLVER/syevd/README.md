# hipSOLVER Symmetric Eigenvalue calculation (divide and conquer algorithm)

## Description

This example illustrates how to calculate the [eigenvalues](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of a [symmetric](https://en.wikipedia.org/wiki/Symmetric_matrix) 3x3 matrix using hipSOLVER.

The eigenvalues of a matrix are defined as such:

$Av_i = \lambda_i v_i$

where

- $A\in\mathbb{R}^{3\times3}$ symmetric matrix,
- $\lambda_i$ for $i\in\{1, 2, 3\}$ eigenvalues (in ascending order),
- $v_i\in\mathbb{R}^3$ eigenvectors corresponding to the $i$-th eigenvalue.

### Application flow

1. Instantiate a vector containing $A$'s 9 elements.
2. Allocate device memory and copy $A$'s elements to the device.
3. Allocate device memory for the outputs of the hipSOLVER function, namely for the calculated eigenvalue vector $W=[\lambda_1, \lambda_2, \lambda_3]$, and the returned `info` value.
4. Create a hipSOLVER handle and query the working space size.
5. Invoke `hipsolverDsyevd` to calculate the eigenvalues of double precision symmetric matrix $A$.
6. Copy the resulting `info` value to the host and check if the operation was successful (indicated by a `0` value).
7. Copy the resulting eigenvalues vector to the host. Print their values and check if their values match the expected.
8. Free all allocated resources.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER is initialized by calling `hipsolverCreate(hipsolverHandle_t*)` and it is terminated by calling `hipsolverDestroy(hipsolverHandle_t)`.

- `hipsolver[SD]syevd` computes the eigenvalues of an $n \times n$ matrix $A$. The correct function signature should be chosen based on the datatype of the input matrix:

  - `S` (single-precision real: `float`)
  - `D` (double-precision real: `double`)

  A complex version of this function is also available under the name `hipsolver[CZ]heevd`. It accepts the same parameters as `hipsolver[SD]syevd`, except that the correct function signature should be chosen based on the following data types:

  - `C` (single-precision complex: `hipFloatComplex`).
  - `Z` (double-precision complex: `hipDoubleComplex`).

  In this example, a double-precision real input matrix is used, in which case the function accepts the following parameters:

  - `hipsolverHandle_t handle`
  - `hipsolverEigMode_t jobz`: Specifies whether the eigenvectors should also be calculated besides the eigenvalues. The following values are accepted:

    - `HIPSOLVER_EIG_MODE_NOVECTOR`: Calculate the eigenvalues only.
    - `HIPSOLVER_EIG_MODE_VECTOR`: Calculate both the eigenvalues and the eigenvectors. The eigenvectors are calculated by a divide and conquer algorithm and are written to the memory location specified by `*A`.

  - `hipSolverFillMode_t uplo`: Specifies whether the upper or lower triangle of the symmetric matrix is stored. The following values are accepted:

    - `HIPSOLVER_FILL_MODE_UPPER`: The provided `*A` pointer points to the upper triangle matrix data.
    - `HIPSOLVER_FILL_MODE_LOWER`: The provided `*A` pointer points to the lower triangle matrix data.

  - `int n`: Number of rows and columns of $A$.
  - `double *A`: Pointer to matrix $A$ in device memory.
  - `int lda`: Leading dimension of matrix $A$.
  - `double *D`: Pointer to array $W$, where the resulting eigenvalues are written.
  - `double *work`: Pointer to working space in device memory.
  - `int lwork`: Size of the working space.
  - `int *devInfo`: Convergence result of the function in device memory. If 0, the algorithm converged, if greater than 0 then `devInfo` elements of the intermediate tridiagonal matrix did not converge to 0. Also, for CUDA backend, if `devInfo = -i` for $0 < i \leq n$, then the the $i^{th}$ parameter is wrong (not counting the handle).

  Return type: `hipsolverStatus_t`.

- `hipsolver[SD]syevd_bufferSize` allows to obtain the size (in bytes) needed for the working space for the `hipsolver[SD]syevd` function. The character matched in `[SD]` coincides with the one in `hipsolver[SD]syevd`.

  This function accepts the following input parameters:

  - `hipsolverHandle_t handle`
  - `hipsolverEigMode_t jobz`: Specifies whether the eigenvectors should also be calculated besides the eigenvalues.
  - `hipSolverFillMode_t uplo`: Specifies whether the upper or lower triangle of the symmetric matrix is stored.
  - `int n`: Number of rows and columns of $A$.
  - `double *A`: Pointer to matrix $A$ in device memory.
  - `int lda`: Leading dimension of matrix $A$.
  - `double *D`: Pointer to array $W$ in device memory, where the resulting eigenvalues are written.
  - `int *lwork`: The required buffer size is written to this location.

  Return type: `hipsolverStatus_t`.

## Used API surface

### hipSOLVER

- `hipsolverCreate`
- `hipsolverDsyevd_bufferSize`
- `hipsolverDsyevd`
- `hipsolverDestroy`
- `HIPSOLVER_EIG_MODE_NOVECTOR`
- `HIPSOLVER_FILL_MODE_UPPER`

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
