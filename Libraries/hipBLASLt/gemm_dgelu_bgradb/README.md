# hipBLASLt GEMM with GELU Derivative and Bias Gradient Reduction

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication with GELU derivative computation and bias gradient reduction.

The operation calculates the following with GELU derivative and bias gradient computation:

$D = \alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C$

$D = D \odot \text{dgelu}(aux\_buffer)$ (element-wise GELU derivative)

$bias\_gradient = \sum_{i=0}^{m-1} D_{i,:}$ (reduction along rows)

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $aux\_buffer$ is an auxiliary input buffer containing values for GELU derivative computation
- $\text{dgelu}(x)$ is the derivative of the GELU activation function
- $bias\_gradient$ is a vector of length $m$ containing the accumulated gradients for bias terms
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create matrix layout descriptors for all matrices (A, B, C, D).
6. Configure batch processing parameters if batch count > 1.
7. Create matrix multiplication descriptor and set operation attributes.
8. Configure DGELU_BGRAD epilogue for combined GELU derivative and bias gradient computation.
9. Allocate and initialize auxiliary buffer with input values for GELU derivative.
10. Set auxiliary buffer attributes including pointer, leading dimension, and batch stride.
11. Configure bias tensor for gradient accumulation.
12. Set up matrix multiplication preferences including workspace size.
13. Query heuristic algorithms to find optimal implementation.
14. Perform matrix multiplication with integrated GELU derivative and bias gradient computation.
15. Copy the result matrix from device to host memory.
16. Clean up auxiliary buffer, bias allocation, and hipBLASLt descriptors.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutSetAttribute()`: Configures properties for batched GEMM, such as batch count (`HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT`) and the stride between matrices (`HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`).
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation. For the GELU derivative and bias gradient computation, the epilogue is set to `HIPBLASLT_EPILOGUE_DGELU_BGRAD`. The auxiliary buffer for the GELU derivative is set with `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`, and the bias pointer and data type are configured using `HIPBLASLT_MATMUL_DESC_BIAS_POINTER` and `HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the matrix multiplication with the GELU derivative and bias gradient computation using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtEpilogue_t`: Defines the epilogue operation, such as `HIPBLASLT_EPILOGUE_DGELU_BGRAD`.

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
- `hipMemcpy`
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
- `HIPBLASLT_EPILOGUE_DGELU_BGRAD`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE`
- `HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`
- `HIPBLASLT_MATMUL_DESC_BIAS_POINTER`
