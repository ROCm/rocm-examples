# rocBLAS Level 1 NRM2 Example

## Description

This example showcases the usage of rocBLAS' Level 1 NRM2 function. NRM2 is an Euclidean norm operator applied on $x$ vector defined as $\sqrt \left( \sum_i{x_i^2} \right)$.

### Application flow

1. Read in and parse command line parameters.
2. Allocate and initialize host vector.
3. Compute CPU reference result.
4. Create a rocBLAS handle.
5. Allocate and initialize device vector.
6. Use the rocBLAS handle to enable passing the `h_result` parameter via a pointer to host memory.
7. Call `rocblas_snrm2()` asynchronous rocBLAS Euclidean norm function.
8. Copy the result from device to host.
9. Destroy the rocBLAS handle, release device memory.
10. Validate the output by comparing it to the CPU reference result.

### Command line interface

The application provides the following optional command line arguments:

- `-x` or `--incx`. The stride between consecutive values in the data array that makes up vector $x$, must be greater than zero. Its default value is 1.
- `-n`. The number of elements in vectors $x$. Its default value is 5.

## Key APIs and Concepts

- rocBLAS is initialized by calling `rocblas_create_handle(rocblas_handle*)` and it is terminated by calling `rocblas_destroy_handle(rocblas_handle)`.

- The _pointer mode_ controls whether scalar parameters must be allocated on the host (`rocblas_pointer_mode_host`) or on the device (`rocblas_pointer_mode_device`). It is controlled by `rocblas_set_pointer_mode`.

- `rocblas_[sdhcz]nrm2` computes the Euclidean norm of a vector as defined above. Depending on the character matched in `[sdhcz]`, the norm can be obtained with different precisions:

  - `s` (single-precision: `rocblas_float`)
  - `d` (double-precision: `rocblas_double`)
  - `h` (half-precision: `rocblas_half`)
  - `c` (single-precision complex: `rocblas_complex`)
  - `z` (double-precision complex: `rocblas_double_complex`).

## Demonstrated API Calls

### rocBLAS

- `rocblas_create_handle`
- `rocblas_destroy_handle`
- `rocblas_handle`
- `rocblas_int`
- `rocblas_set_pointer_mode`
- `rocblas_pointer_mode_host`
- `rocblas_snrm2`
- `rocblas_status`
- `rocblas_status_success`
- `rocblas_status_to_string`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
