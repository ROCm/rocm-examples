# hipBLASLt Batched General Matrix Multiplication (GEMM)

## Description

This example illustrates the use of the `hipBLASLt` library for batched general matrix multiplication.

The operation calculates the following product for multiple matrix sets:

$D_i = \alpha \cdot op_A(A_i) \cdot op_B(B_i) + \beta \cdot C_i$

where $i = 0, 1, ..., batch\_count - 1$

- $\alpha$ and $\beta$ are scalars
- $A_i$ is the $i$-th matrix of dimensions $m \times k$ in the batch
- $B_i$ is the $i$-th matrix of dimensions $k \times n$ in the batch
- $C_i$ and $D_i$ are the $i$-th matrices of dimensions $m \times n$ in the batch
- $op_A(A_i)$ and $op_B(B_i)$ are the result of applying to matrices $A_i$ and $B_i$.

## Application flow

1. Set up matrix dimensions and batch count, allocate memory for input and output matrix batches.
2. Initialize input matrix batches with random values using the `runner` utility class.
3. Copy input matrix batches from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create matrix layout descriptors for all matrices (A, B, C, D).
6. Configure batch processing parameters including batch count and stride offsets.
7. Create matrix multiplication descriptor and set operation attributes.
8. Set up matrix multiplication preferences including workspace size.
9. Query heuristic algorithms to find optimal implementation for batched operations.
10. Perform batched matrix multiplication using the selected algorithm.
11. Copy the result matrix batches from device to host memory.
12. Clean up hipBLASLt descriptors and device allocations.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutSetAttribute()`: Configures properties for batched GEMM, such as batch count (`HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT`) and the stride between matrices (`HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`).
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation, including matrix transformations and epilogue operations.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the batched matrix multiplication using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).

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
- `HIPBLASLT_EPILOGUE_DEFAULT`
- `HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT`
- `HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`
