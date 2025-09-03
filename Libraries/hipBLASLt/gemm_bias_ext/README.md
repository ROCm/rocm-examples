# hipBLASLt Extension API - GEMM with Bias Addition

## Description

This example illustrates the use of the `hipBLASLt` extension API for general matrix multiplication with bias addition.

The operation calculates the following with bias addition:

$D = \alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C + bias$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$ (note: B is transposed in this example)
- $C$ and $D$ are matrices of dimensions $m \times n$
- $bias$ is a vector that is broadcasted and added to the result
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Configure bias information using `set_bias_info()` for bias vector allocation.
4. Copy input matrices from host to device memory.
5. Set up hipBLASLt handle and stream.
6. Create GEMM preference object and set workspace size.
7. Create GEMM object with 16-bit floating point data types and 32-bit float compute type.
8. Configure GEMM epilogue for BIAS operation with bias data type.
9. Allocate and initialize bias vector with example values.
10. Configure GEMM inputs including matrices and bias vector.
11. Set problem dimensions and configure the GEMM operation with BIAS epilogue.
12. Query heuristic algorithms to find optimal implementation.
13. Initialize GEMM with selected algorithm and workspace.
14. Execute the GEMM operation with integrated bias addition.
15. Copy the result matrix from device to host memory.
16. Clean up device allocations and destroy hipBLASLt handle.

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
  - `hipblaslt_ext::GemmEpilogue`: Configures the epilogue operation. `setMode()` is used to set the epilogue to `HIPBLASLT_EPILOGUE_BIAS`, and `setBiasDataType()` sets the data type of the bias vector.
  - `hipblaslt_ext::GemmInputs`: Specifies the input matrices, scalars, and bias vector for the GEMM operation.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose) and `HIPBLAS_OP_T` (transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtEpilogue_t`: Defines the epilogue operation, such as `HIPBLASLT_EPILOGUE_BIAS`.

## Demonstrated API Calls

### hipBLASLt Extension API

- `hipblaslt_ext::Gemm` (constructor)
- `hipblaslt_ext::GemmPreference`
- `hipblaslt_ext::GemmPreference::setMaxWorkspaceBytes`
- `hipblaslt_ext::GemmEpilogue`
- `hipblaslt_ext::GemmEpilogue::setMode`
- `hipblaslt_ext::GemmEpilogue::setBiasDataType`
- `hipblaslt_ext::GemmInputs`
- `hipblaslt_ext::GemmInputs::setA`
- `hipblaslt_ext::GemmInputs::setB`
- `hipblaslt_ext::GemmInputs::setC`
- `hipblaslt_ext::GemmInputs::setD`
- `hipblaslt_ext::GemmInputs::setAlpha`
- `hipblaslt_ext::GemmInputs::setBeta`
- `hipblaslt_ext::GemmInputs::setBias`
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
- `HIPBLAS_OP_T`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLASLT_EPILOGUE_BIAS`
