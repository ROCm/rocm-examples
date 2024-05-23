# rocBLAS Level 1 Swap Example

## Description

This example shows the use of the rocBLAS Level 1 swap operation, which exchanges elements between two HIP vectors. The Level 1 API defines operations between vectors.

### Application flow

1. Read in command-line parameters.
2. Allocate and initialize host vectors.
3. Compute CPU reference result.
4. Create a rocBLAS handle.
5. Allocate and initialize device vectors.
6. Invoke the rocBLAS swap function.
7. Copy the result from device to host.
8. Destroy the rocBLAS handle, release device memory.
9. Validate the output by comparing it to the CPU reference result.

### Command line interface

The application provides the following optional command line arguments:

- `-x` or `--incx`. The stride between consecutive values in the data array that makes up vector $x$, which must be greater than 0. Its default value is 1.

- `-y` or `--incy`. The stride between consecutive values in the data array that makes up vector $y$, which must be greater than 0. Its default value is 1.

- `-n` or `--n`. The number of elements in vectors $x$ and $y$, which must be greater than 0. Its default value is 5.

## Key APIs and Concepts

- rocBLAS is initialized by calling `rocblas_create_handle(rocblas_handle*)` and it is terminated by calling `rocblas_destroy_handle(rocblas_handle)`.

- `rocblas_set_vector(n, elem_size, *x, incx, *y, incy)` is used to copy vectors from host to device memory. `n` is the total number of elements that should be copied, and `elem_size` is the size of a single element in bytes. The elements are copied from `x` to `y`, where the step size between consecutive elements of `x` and `y` is given respectively by `incx` and `incy`. Note that the increment is given in elements, not bytes. Additionally, the step size of either `x`, `y`, or both may also be negative. In this case care must be taken that the correct pointer is passed to `rocblas_set_vector`, as it is not automatically adjusted to the end of the input vector. When `incx` and `incy` are 1, calling this function is equivalent to `hipMemcpy(y, x, n * elem_size, hipMemcpyHostToDevice)`. See the following diagram , which illustrates `rocblas_set_vector(3, sizeof(T), x, incx, y, incy)`:

    ![An illustration of rocblas_set_vector execution.](set_get_vector.svg)

- `rocblas_get_vector(n, elem_size, *x, incx, *y, incy)` is used to copy vectors from device to host memory. Its arguments are similar to `rocblas_set_vector`. Elements are also copied from `x` to `y`.

- `rocblas_Xswap(handle, n, *x, incx, *y, incy)` exchanges elements between vectors `x` and `y`. The two vectors are respectively indexed according to the step increments `incx` and `incy` which are each indexed according to step increments `incx` and `incy` similar to `rocblas_set_vector` and `rocblas_get_vector`. `n` gives the amount of elements that should be exchanged. `X` specifies the data type of the operation, and can be one of `s` (single-precision: `rocblas_float`), `d` (double-precision: `rocblas_double`), `h` (half-precision: `rocblas_half`), `c` (single-precision complex: `rocblas_complex`), or `z` (double-precision complex: `rocblas_double_complex`).

## Demonstrated API Calls

### rocBLAS

- `rocblas_create_handle`
- `rocblas_destroy_handle`
- `rocblas_get_vector`
- `rocblas_handle`
- `rocblas_int`
- `rocblas_set_vector`
- `rocblas_sswap`
- `rocblas_status`
- `rocblas_status_success`
- `rocblas_status_to_string`

### HIP runtime

- `hipFree`
- `hipMalloc`
