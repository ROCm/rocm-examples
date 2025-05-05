# hipBLAS Level 2 Hermitian Rank-2 Update Example

## Description

This example showcases the usage of the hipBLAS Level2 Hermitian rank-2 update functionality. The hipBLAS HER2 function performs a Hermitian rank-2 update operation, which is defined as follows:

$A = A + \alpha\cdot x\cdot y^H + \bar\alpha \cdot y \cdot x^H$,

where $A$ is an $n \times n$ Hermitian complex matrix, $x$ and $y$ are complex vectors of $n$ elements, $\alpha$ is a complex scalar and $v^H$ is the _Hermitian transpose_ of a vector $v \in \mathbb{C}^n$.

### Application flow

1. Read in command-line parameters.
2. Allocate and initialize the host vectors and matrix.
3. Compute CPU reference result.
4. Create a hipBLAS handle.
5. Allocate and initialize the device vectors and matrix.
6. Copy input vectors and matrix from host to device.
7. Invoke the hipBLAS HER2 function.
8. Copy the result from device to host.
9. Destroy the hipBLAS handle and release device memory.
10. Validate the output by comparing it to the CPU reference result.

### Command line interface

The application provides the following optional command line arguments:

- `-a` or `--alpha`. The scalar value $\alpha$ used in the HER2 operation. Its default value is 1.
- `-x` or `--incx`. The stride between consecutive values in the data array that makes up vector $x$, which must be greater than 0. Its default value is 1.
- `-y` or `--incy`. The stride between consecutive values in the data array that makes up vector $y$, which must be greater than 0. Its default value is 1.
- `-n` or `--n`. The dimension of matrix $A$ and vectors $x$ and $y$, which must be greater than 0. Its default value is 5.

## Key APIs and Concepts

- hipBLAS is initialized by calling `hipblasCreate(hipblasHandle_t*)` and it is terminated by calling `hipblasDestroy(hipblasHandle_t)`.
- The _pointer mode_ controls whether scalar parameters must be allocated on the host (`HIPBLAS_POINTER_MODE_HOST`) or on the device (`HIPBLAS_POINTER_MODE_DEVICE`). It is controlled by `hipblasSetPointerMode`.
- `hipblasSetVector` and `hipblasSetMatrix` are helper functions provided by the hipBLAS API for writing data to the GPU, whereas `hipblasGetVector` and `hipblasGetMatrix` are intended for retrieving data from the GPU. Note that `hipMemcpy` can also be used to copy/get data to/from the GPU in the usual way.
- `hipblas[CZ]her2(handle, uplo, n, *alpha, *x, incx, *y, incy, *AP, lda)` computes a Hermitian rank-2 update. The character matched in `[CZ]` denotes the data type of the operation, and can be either `C` (complex float: `hipComplex`), or `Z` (complex double: `hipDoubleComplex`). The required arguments come as the following:
  - `handle`, the hipBLAS API handle.
  - `uplo`. Because a Hermitian matrix is symmetric over the diagonal, except that the values in the upper triangle are the complex conjugate of the values in the lower triangle, the required work can be reduced by only updating a single half of the matrix. The part of the matrix to update is given by `uplo`: `HIPBLAS_FILL_MODE_UPPER` (used in this example) indicates that the upper triangle of $A$ should be updated, `HIPBLAS_FILL_MODE_LOWER` indicates that the lower triangle of $A$ should be updated and `HIPBLAS_FILL_MODE_FULL` indicates that the full matrix will be updated.
  - `n` gives the dimensions of the vector and matrix inputs.
  - `alpha` is the complex scalar.
  - `x` and `y` are the input vectors, and `incx` and `incy` are the increments in elements between items of $x$ and $y$, respectively.
  - `AP` is the device pointer to matrix $A$ in device memory.
  - `lda` is the _leading dimension_ of $A$, that is, the number of elements between the starts of the columns of $A$. Note that hipBLAS matrices are laid out in _column major_ ordering.

- If `ROCM_MATHLIBS_API_USE_HIP_COMPLEX` is defined (adding `#define ROCM_MATHLIBS_API_USE_HIP_COMPLEX` before `#include <hipblas/hipblas.h>`), the hipBLAS API is exposed as using the hip defined complex types. That is, `hipComplex` is a typedef of `hipFloatComplex` (also named `hipComplex`) and they can be used equivalently.
- `hipFloatComplex` and `std::complex<float>` have compatible memory layout, and performing a memory copy between values of these types will correctly perform the expected copy.
- `hipCaddf(a, b)` adds `hipFloatComplex` values `a` and `b` element-wise together. This function is from a family of host/device HIP functions which operate on complex values.

## Demonstrated API Calls

### hipBLAS

- `HIPBLAS_FILL_MODE_UPPER`
- `HIPBLAS_POINTER_MODE_HOST`
- `hipblasCher2`
- `hipComplex`
- `hipblasCreate`
- `hipblasDestroy`
- `hipblasFillMode_t`
- `hipblasHandle_t`
- `hipblasSetMatrix`
- `hipblasSetPointerMode`
- `hipblasSetVector`

### HIP runtime

- `ROCM_MATHLIBS_API_USE_HIP_COMPLEX`
- `hipCaddf`
- `hipFloatComplex`
- `hipFree`
- `hipMalloc`
