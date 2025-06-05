# hipBLAS Level 1 Scal Example

## Description

This example showcases the usage of hipBLAS' Level 1 SCAL function. The Level 1 API defines operations between vector and vector. SCAL is a scaling operator for an $x$ vector defined as $x_i := \alpha \cdot x_i$.

### Application flow

1. Read in and parse command line parameters.
2. Allocate and initialize host vector.
3. Compute CPU reference result.
4. Create a hipBLAS handle.
5. Allocate and initialize device vector.
6. Use the hipBLAS handle to enable passing the $a$ scaling parameter via a pointer to host memory.
7. Call hipBLAS' SCAL function.
8. Copy the result from device to host.
9. Destroy the hipBLAS handle, release device memory.
10. Validate the output by comparing it to the CPU reference result.

### Command line interface

The application provides the following optional command line arguments:

- `-a` or `--alpha`. The scalar value $a$ used in the SCAL operation. Its default value is 3.
- `-x` or `--incx`. The stride between consecutive values in the data array that makes up vector $x$, must be greater than zero. Its default value is 1.
- `-n` or `--n`. The number of elements in vector $x$, must be greater than zero. Its default value is 5.

## Key APIs and Concepts

- hipBLAS is initialized by calling `hipblasCreate(hipblasHandle_t *handle)` and it is terminated by calling `hipblasDestroy(hipblasHandle_t handle)`.
- The _pointer mode_ controls whether scalar parameters must be allocated on the host (`HIPBLAS_POINTER_MODE_HOST`) or on the device (`HIPBLAS_POINTER_MODE_DEVICE`). It is controlled by `hipblasSetPointerMode`.
- `hipblas[SDCZ]scal` multiplies each element of the vector by a scalar. Depending on the character matched in `[SDCZ]`, the scaling can be obtained with different precisions:
  - `S` (single-precision: `float`)
  - `D` (double-precision: `double`)
  - `C` (single-precision complex: `hipComplex`)
  - `Z` (double-precision complex: `hipDoubleComplex`).

## Demonstrated API Calls

### hipBLAS

- `hipblasCreate`
- `hipblasDestroy`
- `hipblasHandle_t`
- `hipblasSetPointerMode`
- `HIPBLAS_POINTER_MODE_HOST`
- `hipblasSscal`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
