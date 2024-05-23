# hipSOLVER Symmetric Eigenvalue via Generalized Jacobi Example

## Description

This example illustrates how to compute the eigenvalues $W$ and eigenvectors $V$ from a symmetric $n \times n$ real matrix $A$ using the Jacobi method.
For computing eigenvalues and eigenvectors of Hermitian (complex) matrices, refer to `hipsolver[CZ]heevj`.

Matrix $A$ is symmetric if $a_{ij} = a_{ji}$.
The results are the eigenvalues $W$ and orthonormal eigenvectors $V$, meaning that the eigenvectors are _orthogonal_ to each other and are _normalized_.

The results are verified by filling in the equation we wanted to solve:
$A \underset{\text{right}}{\times} V = V \times W$ and checking the error.

### Command line interface

The application has an optional argument:

- `-n <n>` with size of the $n \times n$ matrix $A$. The default value is `3`.

## Application flow

1. Parse command line arguments for dimensions of the input matrix.
2. Declare the host side inputs and outputs.
3. Initialize a random symmetric $n \times n$ input matrix.
4. Set the solver parameters.
5. Allocate device memory and copy input matrix from host to device.
6. Initialize hipSOLVER.
7. Query the required working space and allocate this on device.
8. Compute the eigenvector and eigenvalues.
9. Retrieve the results by copying from device to host.
10. Print the results
11. Validate the results
12. Free the memory allocations on device.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER (`hipsolverHandle_t`) gets initialized by `hipsolverCreate` and destroyed by `hipsolverDestroy`.
- `hipsolverEigMode_t`: specifies whether only the eigenvalues or also the eigenvectors should be computed. Passed to `hipsolverDsyevj` as `jobz`.

  - `HIPSOLVER_EIG_MODE_VECTOR`: compute the eigenvalues and eigenvectors.
  - `HIPSOLVER_EIG_MODE_NOVECTOR`: only compute the eigenvalues.

- `hipsolverFillMode_t`: specifies which part of $A$ to use.
  - `HIPSOLVER_FILL_MODE_LOWER`: data is stored in the lower triangle of $A$ .
  - `HIPSOLVER_FILL_MODE_UPPER`: data is stored in the upper triangle of $A$ .

- `hipsolverCreateSyevjInfo`: initializes a structure for the parameters and results for calling `syevj`.
- `hipsolverDestroySyevjInfo`: destroys the structure for the parameters and results for calling `syevj`.
- `hipsolverXsyevjSetMaxSweeps`: configures the max amounts of sweeps
- `hipsolverXsyevjSetTolerance`: configures  the tolerance of `syevj`.
- `hipsolverXsyevjSetSortEig` : configures whether to sort the results or not
- `hipsolver[SD]sygvj_bufferSize` computes the required buffersize `lwork` from a given configuration.
- `hipsolver[SD]syevj` computes the eigenvalue and optional eigenvector.

  - There are 2 different function signatures depending on the type of the input matrix:

    - `S` single-precision real (`float`)
    - `D` double-precision real (`double`)

    For single- and double-precision complex values, the function `hipsolver[CZ]heevj(...)` is available in hipSOLVER.

    For example, `hipsolverDsyevj(...)` works on `double`s. For the complex datatypes see `hipsolver[CZ]heevj`.

  - `hipsolverHandle_t handle`: hipSOLVER handle, see `hipsolverCreate`
  - `hipsolverEigMode_t jobz`: eigenvector output mode, see `hipsolverEigMode_t`.
  - `hipsolverFillMode_t uplo`: fill mode of $A$, see `hipsolverFillMode_t`.
  - `int n`: number of columns of $A$.
  - `double* A`: pointer to the input matrix $A$ on device memory.
  - `int lda`: leading dimension of $A$.
  - `double* W`: pointer to the output $W$ for the eigenvalues.
  - `double* work`: pointer to the working space.
  - `int lwork`: size of the working space.
  - `int* info`: pointer to write the convergence result to.
  - `syevjInfo_t params`: the structure for the parameters and results for `syevj`.
- `hipsolverXsyevjGetSweeps`: gets the amount of executed sweeps of `syevj`.
- `hipsolverXsyevjGetResidual`: gets the residual of `syevj`.

## Used API surface

### hipSOLVER

- `hipsolverCreate`
- `hipsolverDestroy`
- `hipsolverCreateSyevjInfo`
- `hipsolverDestroySyevjInfo`
- `hipsolverDsyevj`
- `hipsolverDsyevj_bufferSize`
- `hipsolverXsyevjGetResidual`
- `hipsolverXsyevjGetSweeps`
- `hipsolverXsyevjSetMaxSweeps`
- `hipsolverXsyevjSetSortEig`
- `hipsolverXsyevjSetTolerance`
- `hipsolverEigMode_t`
- `hipsolverFillMode_t`
- `hipsolverHandle_t`
- `hipsolverSyevjInfo_t`
- `HIPSOLVER_EIG_MODE_VECTOR`
- `HIPSOLVER_FILL_MODE_LOWER`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
