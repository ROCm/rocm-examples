# hipBLASLt Extension API - GEMM with GELU Derivative and Bias Gradient Reduction

## Description

This example illustrates the use of the `hipBLASLt` extension API for general matrix multiplication with GELU derivative computation and bias gradient reduction.

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
3. Configure bias information using `set_bias_info()` for bias vector allocation.
4. Copy input matrices from host to device memory.
5. Set up hipBLASLt handle and stream.
6. Create GEMM preference object and set workspace size.
7. Create GEMM object with 16-bit floating point data types and 32-bit float compute type.
8. Configure GEMM epilogue for DGELU_BGRAD operation with bias data type and auxiliary buffer attributes.
9. Allocate and initialize auxiliary buffer with input values for GELU derivative computation.
10. Configure GEMM inputs including matrices, bias vector, and auxiliary buffer.
11. Set problem dimensions and configure the GEMM operation with DGELU_BGRAD epilogue.
12. Query heuristic algorithms to find optimal implementation.
13. Initialize GEMM with selected algorithm and workspace.
14. Execute the GEMM operation with integrated GELU derivative and bias gradient computation.
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
  - `hipblaslt_ext::GemmEpilogue`: Configures the epilogue operation. `setMode()` is used to set the epilogue to `HIPBLASLT_EPILOGUE_DGELU_BGRAD`, and `setBiasDataType()` sets the data type of the bias vector. The auxiliary buffer is configured with `setAuxLeadingDimension()` and `setAuxBatchStride()`.
  - `hipblaslt_ext::GemmInputs`: Specifies the input matrices, scalars, bias vector, and auxiliary buffer for the GEMM operation.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtEpilogue_t`: Defines the epilogue operation, such as `HIPBLASLT_EPILOGUE_DGELU_BGRAD`.

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
- `HIPBLASLT_EPILOGUE_DGELU_BGRAD`
