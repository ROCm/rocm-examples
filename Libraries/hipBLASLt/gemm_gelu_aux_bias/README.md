# hipBLASLt GEMM with GELU Activation, Auxiliary Output, and Bias Addition

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication with GELU activation, auxiliary output storage, and bias addition.

The operation calculates the following with GELU activation and bias addition:

$D = \text{GELU}(\alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C + bias)$

$aux\_buffer = \alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C + bias$ (pre-activation values)

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $bias$ is a vector that is broadcasted and added to the result
- $\text{GELU}(x)$ is the Gaussian Error Linear Unit activation function
- $aux\_buffer$ stores the pre-activation values for use in backpropagation
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create matrix layout descriptors for all matrices (A, B, C, D).
6. Configure batch processing parameters if batch count > 1.
7. Create matrix multiplication descriptor and set operation attributes.
8. Configure GELU_AUX_BIAS epilogue for combined GELU activation, auxiliary output, and bias addition.
9. Allocate and configure auxiliary buffer for storing pre-activation values.
10. Set auxiliary buffer attributes including pointer, leading dimension, and batch stride.
11. Allocate and initialize bias vector with example values.
12. Configure bias tensor with data type and pointer.
13. Set up matrix multiplication preferences including workspace size.
14. Query heuristic algorithms to find optimal implementation.
15. Perform matrix multiplication with integrated GELU activation, auxiliary output, and bias addition.
16. Copy the result matrix from device to host memory.
17. Clean up auxiliary buffer, bias allocation, and hipBLASLt descriptors.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutSetAttribute()`: Configures properties for batched GEMM, such as batch count (`HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT`) and the stride between matrices (`HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`).
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation. For the GELU activation with auxiliary output and bias, the epilogue is set to `HIPBLASLT_EPILOGUE_GELU_AUX_BIAS`. The auxiliary buffer is set with `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`, and the bias pointer and data type are configured using `HIPBLASLT_MATMUL_DESC_BIAS_POINTER` and `HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the matrix multiplication with the GELU activation, auxiliary output, and bias addition using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtEpilogue_t`: Defines the epilogue operation, such as `HIPBLASLT_EPILOGUE_GELU_AUX_BIAS`.

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
- `HIPBLASLT_EPILOGUE_GELU_AUX_BIAS`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE`
- `HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`
- `HIPBLASLT_MATMUL_DESC_BIAS_POINTER`
