# hipBLASLt Extension API - Batched General Matrix Multiplication (GEMM)

## Description

This example illustrates the use of the `hipBLASLt` extension API for batched general matrix multiplication.

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
5. Create GEMM preference object and set workspace size.
6. Create GEMM object with data types and compute type.
7. Set up GEMM epilogue (default operation).
8. Configure GEMM inputs including matrix batch pointers.
9. Set problem dimensions including batch count and configure the batched GEMM operation.
10. Query heuristic algorithms to find optimal implementation for batched operations.
11. Initialize GEMM with selected algorithm and workspace.
12. Execute the batched GEMM operation on the specified stream.
13. Copy the result matrix batches from device to host memory.
14. Clean up device allocations and destroy hipBLASLt handle.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **GEMM Object**:
  - `hipblaslt_ext::Gemm`: A C++ class that simplifies batched GEMM operations.
  - `setProblem()`: Configures the GEMM problem dimensions, batch count, epilogue, and inputs.
  - `algoGetHeuristic()`: Queries for a list of high-performance algorithms.
  - `initialize()`: Initializes the GEMM operation with a selected algorithm and workspace.
  - `run()`: Executes the batched GEMM operation.

- **GEMM Configuration**:
  - `hipblaslt_ext::GemmPreference`: Sets user preferences, such as the maximum workspace size with `setMaxWorkspaceBytes()`.
  - `hipblaslt_ext::GemmInputs`: Specifies the input matrices and scalars for the GEMM operation.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).

## Demonstrated API Calls

### hipBLASLt Extension API

- `hipblaslt_ext::Gemm` (constructor)
- `hipblaslt_ext::GemmPreference`
- `hipblaslt_ext::GemmPreference::setMaxWorkspaceBytes`
- `hipblaslt_ext::GemmEpilogue`
- `hipblaslt_ext::GemmInputs`
- `hipblaslt_ext::GemmInputs::setA`
- `hipblaslt_ext::GemmInputs::setB`
- `hipblaslt_ext::GemmInputs::setC`
- `hipblaslt_ext::GemmInputs::setD`
- `hipblaslt_ext::GemmInputs::setAlpha`
- `hipblaslt_ext::GemmInputs::setBeta`
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
- `hipDataType`
- `hipblasLtHalf`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_16F`
- `HIP_R_32F`
