# hipBLASLt GEMM with TensorFloat-32 (TF32) Precision

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication using TensorFloat-32 (TF32) precision for enhanced performance.

The operation calculates the following matrix product using TF32 acceleration:

$D = \alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- All matrices use 32-bit floating point data type with TF32 compute optimization
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create matrix layout descriptors for all matrices (A, B, C, D) with FP32 data type.
6. Configure batch processing parameters if batch count exceeds 1.
7. Create matrix multiplication descriptor with TF32 compute type.
8. Set transpose operations and epilogue configuration.
9. Set up matrix multiplication preferences including workspace size.
10. Query heuristic algorithms to find optimal TF32-enabled implementation.
11. Perform matrix multiplication using TF32 acceleration.
12. Copy result matrix from device to host memory.
13. Clean up matrix layout descriptors and multiplication descriptor.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision. This example uses `HIPBLAS_COMPUTE_32F_FAST_TF32` to enable TensorFloat-32 acceleration.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation, such as matrix transformations.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the matrix multiplication using TF32 acceleration with a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_32F` for single-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation, such as `HIPBLAS_COMPUTE_32F_FAST_TF32`.

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
- `hipblasLtEpilogue_t`
- `hipblasOperation_t`
- `hipblasComputeType_t`
- `hipDataType`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F_FAST_TF32`
- `HIP_R_32F`
- `HIPBLASLT_EPILOGUE_DEFAULT`
- `HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT`
- `HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`
- `HIPBLASLT_MATMUL_DESC_TRANSA`
- `HIPBLASLT_MATMUL_DESC_TRANSB`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE`
- `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`
