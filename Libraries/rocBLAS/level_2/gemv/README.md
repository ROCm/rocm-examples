# rocBLAS Level 2 General Matrix-Vector Product Example

## Description

This example illustrates the use of the rocBLAS Level 2 General Matrix-Vector Product functionality. This operation implements $y = \alpha \cdot A \cdot x + \beta \cdot y$, where $\alpha$ and $\beta$ are scalars, $A$ is an $m \times n$ matrix, $y$ is a vector of $m$ elements, and $x$ is a vector of $n$ elements. Additionally, this operation may optionally perform a (conjugate) transpose before the multiplication is performed.

### Application flow

1. Read in command-line parameters.
2. Allocate and initialize the host vectors and matrix.
3. Compute CPU reference result.
4. Create a rocBLAS handle.
5. Allocate and initialize the device vectors and matrix.
6. Invoke the rocBLAS GEMV function.
7. Copy the result from device to host.
8. Destroy the rocBLAS handle, release device memory.
9. Validate the output by comparing it to the CPU reference result.

### Command line interface

The application provides the following optional command line arguments:

- `-a` or `--alpha`. The scalar value $\alpha$ used in the GEMV operation. Its default value is 1.
- `-b` or `--beta`. The scalar value $\beta$ used in the GEMV operation. Its default value is 1.
- `-x` or `--incx`. The stride between consecutive values in the data array that makes up vector $x$, which must be greater than 0. Its default value is 1.
- `-y` or `--incy`. The stride between consecutive values in the data array that makes up vector $y$, which must be greater than 0. Its default value is 1.
- `-n` or `--n`. The number of columns in matrix $A$.
- `-m` or `--m`. The number of rows in matrix $A$.

## Key APIs and Concepts

- rocBLAS is initialized by calling `rocblas_create_handle(rocblas_handle*)` and it is terminated by calling `rocblas_destroy_handle(rocblas_handle)`.

- The _pointer mode_ controls whether scalar parameters must be allocated on the host (`rocblas_pointer_mode_host`) or on the device (`rocblas_pointer_mode_device`). It is controlled by `rocblas_set_pointer_mode`.

- `rocblas_Xgemv(handle, trans, m, n, *alpha, *A, lda, *x, incx, *beta, *y, incy)` computes a general matrix-vector product. `m` and `n` specify the dimensions of matrix $A$ _before_ any transpose operation is performed on it. `lda` is the _leading dimension_ of $A$: the number of elements between the starts of columns of $A$. Columns of $A$ are packed in memory. Note that rocBLAS matrices are stored in _column major_ ordering in memory. `x` and `y` specify vectors $x$ and $y$, and `incx` and `incy` denote the increment between consecutive items of the respective vectors in elements. `trans` specifies a matrix operation that may be performed before the matrix-vector product is computed:
  - `rocblas_operation_none` specifies that no operation is performed. In this case, $x$ needs to have $n$ elements, and $y$ needs to have $m$ elements.
  - `rocblas_operation_transpose` specifies that $A$ should be transposed ($op(A) = A^T$) before the matrix-vector product is performed.
  - `rocblas_operation_conjugate_tranpose` specifies that $A$ should be conjugate transposed ($op(A) = A^H$) before the matrix-vector product is performed. In this and the previous case, $x$ needs to have $m$ elements, and $y$ needs to have $n$ elements.
`X` is a placeholder for the data type of the operation and can be either `s` (float: `rocblas_float`) or `d` (double: `rocblas_double`).

## Demonstrated API Calls

### rocBLAS

- `rocblas_create_handle`
- `rocblas_destroy_handle`
- `rocblas_float`
- `rocblas_handle`
- `rocblas_int`
- `rocblas_operation`
- `rocblas_operation_none`
- `rocblas_operation_transpose`
- `rocblas_pointer_mode_host`
- `rocblas_set_pointer_mode`
- `rocblas_sgemv`
- `rocblas_status`
- `rocblas_status_success`
- `rocblas_status_to_string`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
