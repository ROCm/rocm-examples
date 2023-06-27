# rocSOLVER SYEV: Solving Eigenvalue for Symmetric Matrices

## Description

This example illustrates how to compute the eigenvalues $W$ and eigenvectors $V$ from a symmetric $n \times n$ real matrix $A$.

Matrix $A$ is symmetric if $a_{ij} = a_{ji}$.
The results are the eigenvalues $W$ and orthonormal eigenvectors $V$, meaning that the eigenvectors are _orthogonal_ to each other and are _normalized_.

The results are verified by filling in the equation we wanted to solve:
$A \underset{\text{right}}{\times} V = V \times W$ and checking the error.

### Command line arguments

The application has an optional argument:

- `-n <n>` with size of the $n \times n$ matrix $A$. The default value is `3`.

## Application flow

1. Parse command line arguments for dimensions of the input matrix.
2. Declare the host side inputs and outputs.
3. Initialize a random symmetric $n \times n$ input matrix.
4. Set the solver parameters.
5. Allocate device memory and copy input matrix from host to device.
6. Initialize rocBLAS.
7. Allocate the required working space on device.
8. Compute the eigenvector and eigenvalues.
9. Retrieve the results by copying from device to host.
10. Print the results
11. Validate the results
12. Free the memory allocations on device.

## Key APIs and Concepts

### rocSOLVER

- `rocblas_create_handle(...)` initializes rocblas.
- `rocsolver_[sd]syev(...)` computes the eigenvalue and optional eigenvector.
  - There are 2 different function signatures depending on the type of the input matrix:
    - `s` single-precision real (`float`)
    - `d` double-precision real (`double`)

    For single- and double-precision complex values, the function `rocsolver_[cz]heev(...)` is available in rocSOLVER.
- `rocblas_evect`: specifies how the eigenvectors are computed.
  - `rocblas_evect_original`: compute the eigenvalues and eigenvectors
  - `rocblas_evect_tridiagonal`: compute eigenvectors for the symmetric tri-diagonal matrix. However, this is currently not supported.
  - `rocblas_evect_none`: no eigenvectors are computed.
- `rocblas_fill`: specifies which part of $A$ to use.
  - `rocblas_fill_lower`: data is stored in the lower triangle of $A$.
  - `rocblas_fill_upper`: data is stored in the upper triangle of $A$.

## Used API surface

### rocSOLVER

- `rocblas_evect`
- `rocblas_evect_original`
- `rocsolver_dsyev`

### rocBLAS

- `rocblas_create_handle`
- `rocblas_destroy_handle`
- `rocblas_double`
- `rocblas_fill`
- `rocblas_fill_lower`
- `rocblas_handle`
- `rocblas_int`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
