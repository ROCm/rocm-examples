# rocSOLVER Matrix Inversion Example

## Description

This example showcases computing the inversion $A^{-1}$ of a rectangular matrix $n\times n$ matrix $A$ using rocSOLVER. The inversion operation is divided in two steps: first, an [LU factorization](https://en.wikipedia.org/wiki/LU_decomposition) is computed from the input matrix $A$ by use of the `getrf` operation, which yields the lower triangular matrix $L$, upper triangular matrix $U$, and permutation matrix $P$. The results of this operation satisfy $A = PLU$. Next, the inverted matrix $A^{-1}$ is computed from $L$, $U$, and $P$ by using the `getri` operation.

### Application flow

1. Parse command line arguments for the dimension of the input matrix.
2. Declare and initialize a number of constants for the input and output matrix.
3. Allocate and initialize the to-be-inverted input matrix.
4. Allocate device memory and copy input data from host to device.
5. Create a rocBLAS library handle.
6. Invoke the rocSOLVER `getrf` operation with double precision.
7. Copy the `getrf` info output back to the host, and check whether the operation was successful.
8. Invoke the rocSOLVER `getri` operation with double precision.
9. Copy the `getri` info output back to the host, and check again whether the operation was a success.
10. Validate the solution by checking if $A\cdot A^{-1} = I$ holds. This is done by computing $1\cdot A \cdot A^{-1} + -1 \cdot I$ using the `gemm` operation from rocBLAS, and comparing the elements of the result to 0.
11. Free device memory, release rocBLAS handle.

## Key APIs and Concepts

### rocSOLVER

- `rocsolver_[sdcz]getrf` computes the LU-factorization of an $m\times n$ matrix $A$, and optionally also provides a permutation matrix $P$ when partial pivoting is used. Depending on the character matched in `[sdcz]`, the factorization can be computed with different precision:

  - `s` (single-precision: `float`)
  - `d` (double-precision: `double`)
  - `c` (single-precision complex: `rocblas_float_complex`)
  - `z` (double-precision complex: `rocblas_double_complex`).

  Double precision is used in the example. In this case, the function accepts the following parameters:

  - `rocblas_handle handle` is a handle to the rocBLAS library, created using `rocblas_create_handle`.

  - `rocblas_int m` is the number of rows in $A$.

  - `rocblas_int n` is the number of columns in $A$.

  - `double* A` is a device-pointer to the memory of matrix $A$. It should hold at least $n\times lda$ elements. The `getrf` operation is performed in-place, and the resulting $L$ and $U$ matrices are stored in the memory of $A$. Diagonal elements of $L$ are not stored.

  - `rocblas_int lda` is the leading dimension of matrix $A$, which is the stride between the first element of the columns of the matrix. Note that the matrix is laid out in column-major ordering.

  - `rocblas_int* ipiv` is a device-pointer to where the permutation matrix used for partial pivoting is written to. Note that the permutation matrix can be represented using a single array, and so this parameter requires only `min(n, m)` ints of memory. If `ipiv[i] = j`, then row `j` was interchanged with row `i`. Note that row indices are 1-indexed. Partial pivoting enables numerical stability for this algorithm. If this is undesired, the fucntion `rocsolver_[sdcz]getrf_npvt` can be used to omit partial pivoting.

  - `rocblas_int* info` is a device-pointer to a single integer that describes the result of the operation. If `*info` is `0`, the operation was successful. Otherwise `*info` holds the first non-zero pivot (1-indexed), and means that $A$ was not invertible.

  The function returns a `rocblas_status` value, which indicates whether any errors have occurred during the operation.

- `rocblas_[sdcz]getri` inverts a batch of $n\times n$ matrix using the LU-factorization previously obtained using `getrf`. As with the previous operations, different operations for `getri` are available, depending on the character matched in `[sdcz]`.

- `rocblas_[sdcz]getri` inverts an $n\times n$ matrix using the LU-factorization previously obtained using `getrf`. As with `getrf`, `getri` can be computed with a different precision based on the character matched in `[sdcz]`.

  In the case for double precision, the function accepts the following parameters:

  - `rocblas_handle handle` is a handle to the rocBLAS library, created using `rocblas_create_handle`.

  - `rocblas_int n` is the number of rows and columns of the matrix.

  - `double* A` is a device-pointer to an LU-factorized matrix $A$. The matrix should have at least $n\times lda$ elements. On successful exit, the values in this matrix are overwritten with the inversion $A^{-1}$ of the original matrix $A$.

  - `rocblas_int lda` is the leading dimension of $A$, which is the stride between the first element of the columns of the matrix. Note that the matrices are laid out in column-major ordering.

  - `rocblas_int* ipiv` is a device-pointer to the permutation matrix of the LU-factorization of $A$. If no permutation matrix is available, the `rocsolver_[sdcz]_getri_npvt` function can be used instead.

  - `rocblas_int* info` is a device-pointer to a single integer that describes the result of the inversion operation. If `*info` is non-zero, then the inversion failed, and the value indicates the first zero pivot in $A$. If `*info` is zero, then the operation was successful and `A` holds the inverted matrix $A^{-1}$ of the original matrix $A$.

  The function returns a `rocblas_status` value, which indicates whether any errors have occurred during the operation.

### rocBLAS

- `rocblas_[sdcz]gemm` performs a general matrix multiplication in the form of $C = \alpha\cdot op_a(A)\cdot op_b(B) + \beta\cdot C$. This function is showcased in the [rocBLAS level 3 GEMM example](/Libraries/rocBLAS/level_3/gemm/).

## Used API surface

### rocSOLVER

- `rocsolver_dgetrf`
- `rocsolver_dgetri`

### rocBLAS

- `rocblas_create_handle`
- `rocblas_destroy_handle`
- `rocblas_dgemm`
- `rocblas_int`
- `rocblas_handle`
- `rocblas_set_pointer_mode`
- `rocblas_pointer_mode_host`
- `rocblas_operation_none`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToDevice`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
