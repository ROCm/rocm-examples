# hipBLASLt Extension API - GEMM with Split-K Tuning

## Description

This example illustrates the use of the `hipBLASLt` extension API for general matrix multiplication with Split-K tuning optimization.

The operation calculates the following matrix product with Split-K optimization:

$D = \alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- The K dimension is split across multiple GPU thread blocks for improved parallelization
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Query all available algorithms using comprehensive algorithm enumeration.
6. Create GEMM preference object and set workspace size.
7. Create GEMM object with data types and compute type.
8. Set up GEMM epilogue and configure inputs.
9. Set problem dimensions and configure the GEMM operation.
10. Create tuning configurations: default and Split-K with factor 8.
11. Validate algorithm support for both tuning configurations.
12. Allocate additional workspace if Split-K requires more memory.
13. Execute GEMM with default tuning configuration.
14. Re-initialize and execute GEMM with Split-K tuning configuration.
15. Clean up additional workspace allocations if needed.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Algorithm Discovery**:
  - `hipblaslt_ext::getAllAlgos()`: Retrieves all available algorithms for a given GEMM configuration.

- **GEMM Object**:
  - `hipblaslt_ext::Gemm`: A C++ class that simplifies GEMM operations.
  - `setProblem()`: Configures the GEMM problem dimensions, batch count, epilogue, and inputs.
  - `isAlgoSupported()`: Checks if a specific algorithm is supported for the current problem configuration and tuning options, and returns its workspace size.
  - `initialize()`: Initializes the GEMM operation with a selected algorithm, tuning configuration, and workspace.
  - `run()`: Executes the GEMM operation.

- **GEMM Configuration**:
  - `hipblaslt_ext::GemmPreference`: Sets user preferences, such as the maximum workspace size with `setMaxWorkspaceBytes()`.
  - `hipblaslt_ext::GemmInputs`: Specifies the input matrices and scalars for the GEMM operation.
  - `hipblaslt_ext::GemmTuning`: Configures tuning parameters, such as the Split-K factor with `setSplitK()`.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblaslt_ext::GemmType`: Specifies the type of GEMM operation, such as `HIPBLASLT_GEMM`.

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
- `hipblaslt_ext::GemmTuning`
- `hipblaslt_ext::GemmTuning::setSplitK`
- `hipblaslt_ext::Gemm::setProblem`
- `hipblaslt_ext::Gemm::isAlgoSupported`
- `hipblaslt_ext::Gemm::initialize`
- `hipblaslt_ext::Gemm::run`
- `hipblaslt_ext::getAllAlgos`
- `hipblaslt_ext::GemmType::HIPBLASLT_GEMM`

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
- `hipblaslt_ext::GemmType`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLAS_STATUS_SUCCESS`
