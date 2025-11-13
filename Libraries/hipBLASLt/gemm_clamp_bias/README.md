# hipBLASLt General Matrix Multiplication with Clamp Bias

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication with clamp bias functionality.

The operation calculates the following product with bias addition and clamping:

$D = clamp(\alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C + bias, clamp\_lower, clamp\_upper)$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $bias$ is a vector that is broadcasted and added to the result
- $clamp\_lower$ and $clamp\_upper$ define the bounds for clamping the output
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create matrix layout descriptors for all matrices (A, B, C, D).
6. Configure batch processing parameters if batch count > 1.
7. Create matrix multiplication descriptor and set operation attributes.
8. Configure CLAMP_BIAS_EXT epilogue for bias addition with clamping operation.
9. Set clamp bounds using epilogue arguments (lower and upper limits).
10. Allocate and initialize bias vector with example values.
11. Set bias data type and pointer in the matrix multiplication descriptor.
12. Set up matrix multiplication preferences including workspace size.
13. Query heuristic algorithms to find optimal implementation.
14. Perform matrix multiplication with integrated bias addition and clamping.
15. Copy the result matrix from device to host memory.
16. Clean up bias allocation and hipBLASLt descriptors.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutSetAttribute()`: Configures properties for batched GEMM, such as batch count (`HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT`) and the stride between matrices (`HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`).
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation. For clamp bias, the epilogue is set to `HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT`, and the bias pointer, data type, and clamp bounds are configured using appropriate attributes.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Clamp Configuration**:
  - `HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT`: Defines the epilogue operation that adds bias and clamps the result.
  - `HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT`: Sets the lower bound for clamping.
  - `HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT`: Sets the upper bound for clamping.
  - `HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`: Specifies the data type of the bias vector.
  - `HIPBLASLT_MATMUL_DESC_BIAS_POINTER`: Sets the pointer to the bias vector.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the matrix multiplication with bias addition and clamping using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtEpilogue_t`: Defines the epilogue operation, such as `HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT`.

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
- `hipblasLtHalf`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT`
- `HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`
- `HIPBLASLT_MATMUL_DESC_BIAS_POINTER`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT`
