# rocSOLVER Symmetric Eigenvalue Solver for Batched Matrices

## Description

This example illustrates how to solve an eigenvalue problem for a batch $A$ of $m$ symmetric matrices $A_i$ using rocSOLVER. That is, showcases how to compute the eigenvalues and eigenvectors of a batch of symmetric matrices.

Generally, in an eigenvalue problem, we are looking for $\mathbf{x}$ vectors with $\lambda$ scalars that fulfill the

$$
A_i \cdot \mathbf{x} = \lambda \cdot \mathbf{x}
$$

equation.

The solver evaluates the following equation for a batch of $m$ symmetric matrices, named as $A_i$, and with the size of $n \times n$:

$$A_i \cdot V_i = W_i \cdot V_i$$

for each $0 \leq i < m$.

The set of orthonormalized eigenvectors can be settled to a column of a matrix as

$$
V_i = \left[\mathbf{x_{i_0}}, \dots, \mathbf{x_{i_j}}, \dots, \mathbf{x_{i_{n-1}}}\right]
$$

and the eigenvalues as a diagonal matrix:

$$
W_i = \mathrm{diag}\left(\mathbf{w_i}\right) = \mathrm{diag}\left([\lambda_{i_0}, \dots, \lambda_{i_j}, \dots, \lambda_{i_{n-1}}]\right) =
\begin{bmatrix}
\lambda_{i_0} & & & & & \\
 & \lambda_{i_1} & & & & \\
 & & \ddots & & & \\
 & & & \lambda_{i_j} & & \\
 & & & & \ddots & \\
 & & & & & \lambda_{i_{n-1}}
\end{bmatrix}
$$

The solver gives back an array of $V_i$ matrices and the $W$ matrix of the eigenvalues:

$$
W = \left[\mathbf{w_0}, \dots, \mathbf{w_i}, \dots \mathbf{w_{m-1}}\right]
$$

The results are verified, in the example, by filling in the equation we wanted to solve for each matrix of the batch:

$A_i \cdot V_i = V_i \cdot W_i$
and checking the error.

### Command line interface

The application provides the following optional command line arguments:

- `-n <n>` with size of the $n \times n$ matrix $A$. The default value is `3`.
- `-c <c>` the size of the batch. Default value is `3`.

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
10. Free the memory allocations on device.
11. Print the results
12. Validate the results

## Key APIs and Concepts

- The performance of a numerical multi-linear algebra code can be heavily increased by using tensor contractions [ [Y. Shi et al., HiPC, pp 193, 2016.](https://doi.org/10.1109/HiPC.2016.031) ], thereby similarly to other linear algebra libraries like hipBLAS rocSOLVER also has a `_batched` and a `_strided_batched` [ [C. Jhurani and P. Mullowney, JPDP Vol 75, pp 133, 2015.](https://doi.org/10.1016/j.jpdc.2014.09.003) ] extensions.<br/>
We can apply the same operation for several matrices if we combine them into batched matrices. Batched computation has a performance improvement for a large number of small matrices. For a constant stride between matrices, further acceleration is available by strided batched solvers.

### rocSOLVER

- `rocsolver_[sd]syev_batched(...)` computes the eigenvalues and optionally the eigenvectors of a batch of matrices.
  - There are 2 different function signatures depending on the type of the input matrix:
    - `s` single-precision real (`float`)
    - `d` double-precision real (`double`)

    For single- and double-precision complex values, the function `rocsolver_[cz]heev_batched(...)` is available in rocSOLVER.

    In this example a double-precision real input matrix is used, in which case the function accepts the following parameters:
    - `rocblas_handle handle`
    - `const rocblas_evect evect` Specifies whether the eigenvectors should also be calculated besides the eigenvalues. The following values are accepted:
      - `rocblas_evect_original`: Calculate both the eigenvalues and the eigenvectors.
      - `rocblas_evect_none`: Calculate the eigenvalues only.
    - `const rocblas_fill uplo`:  Specifies whether the upper or lower triangle of the symmetric matrix is stored. The following values are accepted:
      - `rocblas_fill_lower`: The provided `*A` pointer points to the lower triangle matrix data.
      - `rocblas_fill_upper`: The provided `*A` pointer points to the upper triangle matrix data.
    - `const rocblas_int n`: Number of rows and columns of $A$.
    - `double* const A[]`: Array of batch $A_i$ matrices in device memory. After execution it contains the eigenvectors, if they were requested and the algorithm converged.
    - `rocblas_int lda`: Leading dimension of matrix $A$ (same for all matrices in the batch). $lda \geq n$.
    - `double* D`:  Pointer to array $\lambda_i$. It is initially used to internally store the leading diagonals of the internal tridiagonal matrices $T_i$ associated with the $A_i$. Eventually this diagonal converges to the resulting eigenvalues.
    - `const rocblas_stride strideD`: Stride from the start of one vector $D_i$ to the next one $D_{j+1}$.
    - `double* E`: This array is used to work internally with the tridiagonal matrices $T_i$ associated with the $A_i$. It stores the super/subdiagonals of these tridiagonal matrices (they are symmetric, so only one of the diagonals is needed).
    - `const rocblas_stride strideE`: Stride from the start of one vector $E_i$ to the next one $E_{i+1}$.
    - `rocblas_int* info`: Array of $m$ integers on the GPU. If `info[i]` = 0, successful exit for matrix $A_i$. If `info[i] > 0`, the algorithm did not converge.
    - `const rocblas_int batch_count`: Number of matrices in the batch.

### rocBLAS

- rocBLAS is initialized by calling `rocblas_create_handle(rocblas_handle t*)` and it is terminated by calling `rocblas_destroy_handle(t)`.

## Used API surface

### rocSOLVER

- `rocblas_evect`
- `rocblas_evect_original`
- `rocsolver_dsyev_batched`

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
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
