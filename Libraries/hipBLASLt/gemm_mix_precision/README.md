# hipBLASLt Mixed Precision General Matrix Multiplication

## Description

This example illustrates the use of the `hipBLASLt` library for mixed precision general matrix multiplication with matrix scaling.

The operation calculates the following product with mixed precision and scaling:

$D = \alpha \cdot (scale_A \cdot op_A(A)) \cdot op_B(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $scale_A$ is a scaling factor applied to matrix A (set to 2.0 in this example)
- $A$ is a matrix of dimensions $m \times k$ stored in 8-bit floating point (E4M3 FNUZ)
- $B$ is a matrix of dimensions $k \times n$ stored in 16-bit floating point
- $C$ and $D$ are matrices of dimensions $m \times n$ stored in 32-bit floating point
- Computation is performed using 32-bit floating point with 16-bit fast mode
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices with different precisions.
2. Initialize input matrices with random values using the `runner` utility class with mixed types.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create matrix layout descriptors with different data types for each matrix.
6. Configure batch processing parameters if batch count > 1.
7. Create matrix multiplication descriptor with fast 16-bit compute type.
8. Allocate and configure scaling factor for matrix A.
9. Set scaling attribute in the matrix multiplication descriptor.
10. Set up matrix multiplication preferences including workspace size.
11. Query heuristic algorithms to find optimal implementation.
12. Perform mixed precision matrix multiplication with scaling.
13. Copy the result matrix from device to host memory.
14. Clean up scaling allocation and hipBLASLt descriptors.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension. This example uses different data types for each matrix (`HIP_R_8F_E4M3_FNUZ`, `HIP_R_16F`, `HIP_R_32F`).
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation, including the scaling factor for matrix A with `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the mixed-precision matrix multiplication using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_8F_E4M3_FNUZ`, `HIP_R_16F`, `HIP_R_32F`).
  - `hipblasComputeType_t`: Sets the precision for the computation, such as `HIPBLAS_COMPUTE_32F_FAST_16F`.

## Demonstrated API Calls

### hipBLASLt

- `hipblasLtMatmul`
- `hipblasLtMatrixLayoutCreate`
- `hipblasLtMatrixLayoutSetAttribute`
- `hipblasLtMatrixLayoutDestroy`
- `hipblasLtMatmulDescCreate`
- `hipblasLtMatmulDescSetAttribute`
- `hipblasLtMatmulDescDestroy`
- `hipblasLtMatmulPreferenceCreate`
- `hipblasLtMatmulPreferenceSetAttribute`
- `hipblasLtMatmulPreferenceDestroy`
- `hipblasLtMatmulAlgoGetHeuristic`
- `hipblasLtCreate`
- `hipblasLtDestroy`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipHostMalloc`
- `hipMalloc`
- `hipMemcpyAsync`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`

### Data Types and Enums

- `hipblasLtHandle_t`
- `hipblasLtMatrixLayout_t`
- `hipblasLtMatmulDesc_t`
- `hipblasLtMatmulPreference_t`
- `hipblasLtMatmulHeuristicResult_t`
- `hipblasOperation_t`
- `hipblasComputeType_t`
- `hipblasLtEpilogue_t`
- `hipDataType`
- `hipblaslt_f8_fnuz`
- `hipblasLtHalf`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F_FAST_16F`
- `HIP_R_8F_E4M3_FNUZ`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLASLT_EPILOGUE_DEFAULT`
- `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`
