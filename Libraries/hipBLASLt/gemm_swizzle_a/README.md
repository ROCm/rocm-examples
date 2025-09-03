# hipBLASLt GEMM with Matrix A Swizzling

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication with matrix A swizzling optimization.

The operation calculates the following matrix product with swizzled memory layout:

$D = \alpha \cdot op_A(A_{swizzled}) \cdot op_B(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $A_{swizzled}$ is matrix A with optimized memory layout (swizzled pattern)
- $A$ is a matrix of dimensions $k \times m$ (with transpose operation)
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Execute three different GEMM configurations for comparison:
   a. **Regular GEMM**: Standard matrix multiplication without swizzling (FP16)
   b. **Swizzled GEMM**: Matrix A swizzling with FP16 data types
   c. **Swizzled F8 GEMM**: Matrix A swizzling with 8-bit floating point data types
3. For each configuration:
   a. Initialize input matrices with random values using runner utility classes
   b. Copy input matrices from host to device memory
   c. Apply swizzling transformation to matrix A when enabled
   d. Execute matrix multiplication with performance timing
   e. Copy results back to host memory
4. Validate results by comparing swizzled outputs with regular GEMM baseline.
5. Display performance metrics and validation results.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutSetAttribute()`: Configures matrix properties. For swizzling, `HIPBLASLT_MATRIX_LAYOUT_ORDER` is set to `HIPBLASLT_ORDER_COL16_4R8` or `HIPBLASLT_ORDER_COL16_4R16` depending on the data type.
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation, such as matrix transformations.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the matrix multiplication using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_T` (transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision, `HIP_R_8F_E4M3_FNUZ` for 8-bit floating point).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtOrder_t`: Defines the memory layout of a matrix, such as `HIPBLASLT_ORDER_COL16_4R8` and `HIPBLASLT_ORDER_COL16_4R16` for swizzling.

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
- `hipEventCreate`
- `hipEventRecord`
- `hipEventElapsedTime`
- `hipEventDestroy`
- `hipFree`
- `hipHostMalloc`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyAsync`
- `hipMemcpyDeviceToHost`
- `hipMemcpyDeviceToDevice`
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
- `hipblasLtOrder_t`
- `hipblasLtEpilogue_t`
- `hipblasOperation_t`
- `hipblasComputeType_t`
- `hipDataType`
- `hipblasLtHalf`
- `hipblaslt_f8_fnuz`
- `HIPBLAS_OP_N`
- `HIPBLAS_OP_T`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_8F_E4M3_FNUZ`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLASLT_ORDER_COL16_4R8`
- `HIPBLASLT_ORDER_COL16_4R16`
- `HIPBLASLT_EPILOGUE_DEFAULT`
- `HIPBLASLT_MATRIX_LAYOUT_ORDER`
- `HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT`
- `HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`
- `HIPBLASLT_MATMUL_DESC_TRANSA`
- `HIPBLASLT_MATMUL_DESC_TRANSB`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE`
- `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`
