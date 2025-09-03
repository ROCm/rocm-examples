# hipBLASLt Extension API - GEMM with Algorithm Selection by Index

## Description

This example illustrates the use of the `hipBLASLt` extension API for general matrix multiplication with algorithm selection by index.

The operation calculates the following matrix product:

$D = \alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create GEMM preference object and set workspace size.
6. Create GEMM object with data types and compute type.
7. Set up GEMM epilogue and configure inputs.
8. Set problem dimensions and configure the GEMM operation.
9. Iteratively search for valid algorithms using index-based algorithm retrieval.
10. For each batch of algorithm indices, check algorithm support and workspace requirements.
11. Select the first valid algorithm that fits within workspace constraints.
12. Display the found algorithm index.
13. Initialize GEMM with the selected algorithm and workspace.
14. Execute the GEMM operation.
15. Clean up device allocations and destroy hipBLASLt handle.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **GEMM Object**:
  - `hipblaslt_ext::Gemm`: A C++ class that simplifies GEMM operations.
  - `setProblem()`: Configures the GEMM problem dimensions, batch count, epilogue, and inputs.
  - `isAlgoSupported()`: Checks if a specific algorithm is supported for the current problem configuration and returns its workspace size.
  - `initialize()`: Initializes the GEMM operation with a selected algorithm and workspace.
  - `run()`: Executes the GEMM operation.

- **Algorithm Selection by Index**:
  - `hipblaslt_ext::getAlgosFromIndex()`: Retrieves algorithm information based on a vector of indices.
  - `hipblaslt_ext::getIndexFromAlgo()`: Retrieves the index of a given algorithm.

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
- `hipblaslt_ext::Gemm::isAlgoSupported`
- `hipblaslt_ext::Gemm::initialize`
- `hipblaslt_ext::Gemm::run`
- `hipblaslt_ext::getAlgosFromIndex`
- `hipblaslt_ext::getIndexFromAlgo`

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
- `HIPBLAS_STATUS_SUCCESS`
- `HIPBLAS_STATUS_INVALID_VALUE`
