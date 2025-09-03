# hipBLASLt Extension API - GEMM Tuning Status Check

## Description

This example illustrates the use of the `hipBLASLt` extension API for checking matrix multiplication tuning status.

The operation checks whether a given matrix multiplication configuration has been tuned for optimal performance:

$D = \alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $k \times m$ (with transpose operation)
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Initialize hipBLASLt handle.
2. Create matrix multiplication descriptor with compute type and data type.
3. Set matrix operation (transpose) for matrix A.
4. Configure pointer mode for alpha and beta parameters.
5. Parse matrix dimensions from command line arguments (default: 128x128x128).
6. Create matrix layout descriptors for all matrices (A, B, C, D).
7. Check tuning status using the extension API.
8. Display whether the configuration is tuned or untuned.
9. Clean up matrix layout descriptors and matrix multiplication descriptor.
10. Destroy hipBLASLt handle.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Tuning Status Check**:
  - `hipblaslt_ext::matmulIsTuned()`: Checks if a specific matrix multiplication configuration has been tuned for optimal performance.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation, such as matrix transformations and pointer modes.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_T` (transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtPointerMode_t`: Defines the location of scalar parameters, such as `HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST`.

## Demonstrated API Calls

### hipBLASLt Extension API

- `hipblaslt_ext::matmulIsTuned`

### hipBLASLt Core API

- `hipblasLtCreate`
- `hipblasLtDestroy`
- `hipblasLtMatmulDescCreate`
- `hipblasLtMatmulDescSetAttribute`
- `hipblasLtMatmulDescDestroy`
- `hipblasLtMatrixLayoutCreate`
- `hipblasLtMatrixLayoutDestroy`

### Data Types and Enums

- `hipblasLtHandle_t`
- `hipblasLtMatmulDesc_t`
- `hipblasLtMatrixLayout_t`
- `hipblasOperation_t`
- `hipblasComputeType_t`
- `hipblasLtPointerMode_t`
- `hipDataType`
- `HIPBLAS_OP_T`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLASLT_MATMUL_DESC_TRANSA`
- `HIPBLASLT_MATMUL_DESC_POINTER_MODE`
- `HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST`
