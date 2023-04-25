# rocSOLVER Symmetric Eigenvalue Calculation for Strided Batched Matrices

## Description

This example illustrates how to solve the standard symmetric-definite eigenvalue problem for a strided batch $A$ of $m$ symmetric matrices $A_i$ using rocSOLVER. That is, showcases how to compute the eigenvalues and eigenvectors of a strided batch of symmetric matrices.

Given a (strided) batch of $m$ symmetric matrices $A_i$ of dimension $n$, the said problem consists on solving the following equation:

$A_i V_i = W_i V_i$

for each $0 \leq i < m$.

A solution for this problem is given by $m$ pairs $(V_i, W_i)$, where
- $V_i$ is an $n \times n$ orthonormal matrix containing (as columns) the eigenvectors $V_{i_j}$ for $j = 0, \dots, n-1$ and
- $\W_i$ is an $n \times n$ diagonal matrix containing the eigenvalues $\lambda_{i_j}$ for $j = 0, \dots, n-1$

such that

$A_i V_{i_j} = W_{i_j} V_{i_j}$

for $j = 0, \dots, n-1$ and $i = 0, \dots, m-1$.

The results are verified by filling in the equation we wanted to solve for each matrix of the batch:

$A_i \cdot V_i = V_i \ctod W_i$ and checking the error.

### Command line interface

The application provides the following optional command line arguments:

- `-n <n>` with size of the $n \times n$ matrix $A$. The default value is `3`.
- `-c <c>` the size of the batch. Default value is `3`.
- `-p <p>` The size of the padding. This value is used to calculate the stride for the input matrix, eigenvalues and the tridiagonal matrix.


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
- The performance of a numerical multi-linear algebra code can be heavily increased by using tensor contractions [ [Y. Shi et al., HiPC, pp 193, 2016.](https://doi.org/10.1109/HiPC.2016.031) ], thereby similarly to other linear algebra libraries like hipBLAS rocSOLVER also has a `_batched` and a `_strided_batched` [ [C. Jhurani and P. Mullowney, JPDP Vol 75, pp 133, 2015.](https://doi.org/10.1016/j.jpdc.2014.09.003) ] extensions.<br/>
We can apply the same operation for several matrices if we combine them into batched matrices. Batched computation has a performance improvement for a large number of small matrices. For a constant stride between matrices, further acceleration is available by strided batched solvers.

### rocSOLVER

- `rocsolver_[sd]syev_strided_batched(...)` computes the eigenvalues and optionally the eigenvectors of a (strided) batch of matrices.
  - There are 2 different function signatures depending on the type of the input matrix:
    - `s` single-precision real (`float`)
    - `d` double-precision real (`double`)

    In this example a double-precision real input matrix is used, in which case the function accepts the following parameters:
    - `rocblas_handle handle` 
    - `rocblas_evect evect` Specifies whether the eigenvectors should also be calculated besides the eigenvalues. The following values are accepted:
      - `rocblas_evect_original`: Calculate both the eigenvalues and the eigenvectors.
      - `rocblas_evect_none`: Calculate the eigenvalues only.
    - `rocblas_fill uplo`:  Specifies whether the upper or lower triangle of the symmetric matrix is stored. The following values are accepted:
      - `rocblas_fill_lower`: The provided `*A` pointer points to the lower triangle matrix data.
      - `rocblas_fill_upper`: The provided `*A` pointer points to the upper triangle matrix data.
    - `rocblas_int n`: Number of rows and columns of $A$.
    - `double* A`: Pointer to the first matrix $A$ in device memory. After execution it contains the eigenvectors, if they were requested and the algorithm converged.
    - `rocblas_int lda`: Leading dimension of matrix $A$ (same for all matrices in the batch).
    - `rocblas_stride strideA`: Stride from the start of one matrix $A_i$ to the next one $A_{i+1}$.
    - `double* D`:  Pointer to array $W$, where the resulting eigenvalues are written.
    - `rocblas_stride strideD`: Stride from the start of one vector $D_i$ to the next one $D_{j+1}$.
    - `double* E`: This array is used to work internally with the tridiagonal matrix $T_i$ associated with $A_i$. 
    - `rocblas_stride strideE`: Stride from the start of one vector $E_i$ to the next one $E_(i+1)$. 
    - `rocblas_int* info`: Array of $m$ integers on the GPU. If `info[i]` = 0, successful exit for matrix $A_i$. If `info[i] > 0`, the algorithm did not converge.
    - `rocblas_int batch_count`: Number of matrices in the batch.

### rocBLAS
- rocBLAS is initialized by calling `rocblas_create_handle(rocblas_handle t*)` and it is terminated by calling `rocblas_destroy_handle(t)`.

## Used API surface

### rocSOLVER

- `rocsolver_dsyev_strided_batched`

### rocBLAS

- `rocblas_create_handle`
- `rocblas_destroy_handle`
- `rocblas_double`
- `rocblas_evect`
- `rocblas_evect::rocblas_evect_original`
- `rocblas_fill`
- `rocblas_fill::rocblas_fill_lower`
- `rocblas_handle`
- `rocblas_int`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
