# rocBLAS Level 1 AXPY Example

## Description

This example showcases the usage of rocBLAS' Level 1 AXPY function. The Level 1 API defines operations between vector and vector. AXPY is the operation $y_i=ax_i+y_i$ for two vectors $x$ and $y$, and a scalar value $a$.

### Application flow

1. Read in command-line parameters.
2. Allocate and initialize host vectors.
3. Compute CPU reference result.
4. Create a rocBLAS handle.
5. Allocate and initialize device vectors.
6. Use the rocBLAS handle to enable passing the $a$ parameter via a pointer to host memory.
7. Call rocBLAS' AXPY function.
8. Copy the result from device to host.
9. Destroy the rocBLAS handle, release device memory.
10. Validate the output by comparing it to the CPU reference result.

### Command line interface

The application provides the following optional command line arguments:

- `-a` or `--alpha`. The scalar value $a$ used in the AXPY operation. Its default value is 1.
- `-x` or `--incx`. The stride between consecutive values in the data array that makes up vector $x$, must be greater than zero. Its default value is 1.
- `-y` or `--incy`. The stride between consecutive values in the data array that makes up vector $y$, must be greater than zero. Its default value is 1.
- `-n` or `--n`. The number of elements in vectors $x$ and $y$, must be greater than zero. Its default value is 5.

## Key APIs and Concepts

- rocBLAS is initialized by calling `rocblas_create_handle(rocblas_handle*)` and it is terminated by calling `rocblas_destroy_handle(rocblas_handle)`.

- The _pointer mode_ controls whether scalar parameters must be allocated on the host (`rocblas_pointer_mode_host`) or on the device (`rocblas_pointer_mode_device`). It is controlled by `rocblas_set_pointer_mode`.

- `rocblas_Xaxpy` computes the AXPY operation as defined above. `X` is one of `s` (single-precision: `rocblas_float`), `d` (double-precision: `rocblas_double`), `h` (half-precision: `rocblas_half`), `c` (single-precision complex: `rocblas_complex`), or `z` (double-precision complex: `rocblas_double_complex`).

## Demonstrated API Calls

### rocBLAS

- `rocblas_create_handle`
- `rocblas_destroy_handle`
- `rocblas_float`
- `rocblas_handle`
- `rocblas_int`
- `rocblas_saxpy`
- `rocblas_set_pointer_mode`
- `rocblas_status`
- `rocblas_status_success`
- `rocblas_status_to_string`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
