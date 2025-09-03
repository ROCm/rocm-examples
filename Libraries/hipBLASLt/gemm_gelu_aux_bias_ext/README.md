# hipBLASLt Extension API - GEMM with GELU Activation, Auxiliary Output, and Bias Addition

## Description

This example illustrates the use of the `hipBLASLt` extension API for general matrix multiplication with GELU activation, auxiliary output storage, and bias addition.

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
3. Configure bias information using `set_bias_info()` for bias vector allocation.
4. Copy input matrices from host to device memory.
5. Set up hipBLASLt handle and stream.
6. Create GEMM preference object and set workspace size.
7. Create GEMM object with 16-bit floating point data types and 32-bit float compute type.
8. Configure GEMM epilogue for GELU_AUX_BIAS operation with bias data type and auxiliary buffer attributes.
9. Allocate auxiliary buffer for storing pre-activation values.
10. Configure GEMM inputs including matrices, bias vector, and auxiliary buffer.
11. Set problem dimensions and configure the GEMM operation with GELU_AUX_BIAS epilogue.
12. Query heuristic algorithms to find optimal implementation.
13. Initialize GEMM with selected algorithm and workspace.
14. Execute the GEMM operation with integrated GELU activation, auxiliary output, and bias addition.
15. Clean up auxiliary buffer and device allocations.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **GEMM Object**:
  - `hipblaslt_ext::Gemm`: A C++ class that simplifies GEMM operations.
  - `setProblem()`: Configures the GEMM problem dimensions, batch count, epilogue, and inputs.
  - `algoGetHeuristic()`: Queries for a list of high-performance algorithms.
  - `initialize()`: Initializes the GEMM operation with a selected algorithm and workspace.
  - `run()`: Executes the GEMM operation.

- **GEMM Configuration**:
  - `hipblaslt_ext::GemmPreference`: Sets user preferences, such as the maximum workspace size with `setMaxWorkspaceBytes()`.
  - `hipblaslt_ext::GemmEpilogue`: Configures the epilogue operation. `setMode()` is used to set the epilogue to `HIPBLASLT_EPILOGUE_GELU_AUX_BIAS`, and `setBiasDataType()` sets the data type of the bias vector. The auxiliary buffer is configured with `setAuxLeadingDimension()` and `setAuxBatchStride()`.
  - `hipblaslt_ext::GemmInputs`: Specifies the input matrices, scalars, bias vector, and auxiliary buffer for the GEMM operation.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtEpilogue_t`: Defines the epilogue operation, such as `HIPBLASLT_EPILOGUE_GELU_AUX_BIAS`.

## Demonstrated API Calls

### hipBLASLt Extension API

- `hipblaslt_ext::Gemm` (constructor)
- `hipblaslt_ext::GemmPreference`
- `hipblaslt_ext::GemmPreference::setMaxWorkspaceBytes`
- `hipblaslt_ext::GemmEpilogue`
- `hipblaslt_ext::GemmEpilogue::setMode`
- `hipblaslt_ext::GemmEpilogue::setBiasDataType`
- `hipblaslt_ext::GemmEpilogue::setAuxLeadingDimension`
- `hipblaslt_ext::GemmEpilogue::setAuxBatchStride`
- `hipblaslt_ext::GemmInputs`
- `hipblaslt_ext::GemmInputs::setA`
- `hipblaslt_ext::GemmInputs::setB`
- `hipblaslt_ext::GemmInputs::setC`
- `hipblaslt_ext::GemmInputs::setD`
- `hipblaslt_ext::GemmInputs::setAlpha`
- `hipblaslt_ext::GemmInputs::setBeta`
- `hipblaslt_ext::GemmInputs::setBias`
- `hipblaslt_ext::GemmInputs::setAux`
- `hipblaslt_ext::Gemm::setProblem`
- `hipblaslt_ext::Gemm::algoGetHeuristic`
- `hipblaslt_ext::Gemm::initialize`
- `hipblaslt_ext::Gemm::run`

### hipBLASLt Core API

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
