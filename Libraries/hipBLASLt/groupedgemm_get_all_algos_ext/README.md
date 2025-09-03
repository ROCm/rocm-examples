# hipBLASLt Extension API - Grouped GEMM with Comprehensive Algorithm Testing

## Description

This example illustrates the use of the `hipBLASLt` extension API for grouped general matrix multiplication with comprehensive algorithm selection.

The operation demonstrates two approaches to grouped matrix multiplication:

1. **Single GroupedGemm**: Standard grouped operation with algorithm validation
2. **Multiple GroupedGemm**: Advanced pattern with multiple grouped instances

$D_i = \alpha_i \cdot op_A(A_i) \cdot op_B(B_i) + \beta_i \cdot C_i$ for $i = 0, 1, ..., N-1$

where

- $\alpha_i$ and $\beta_i$ are scalars for each group
- $A_i$ is a matrix of dimensions $m_i \times k_i$
- $B_i$ is a matrix of dimensions $k_i \times n_i$
- $C_i$ and $D_i$ are matrices of dimensions $m_i \times n_i$
- Each group can have different matrix dimensions
- $op_A(A_i)$ and $op_B(B_i)$ are the result of applying to matrices $A_i$ and $B_i$.

## Application flow

1. Set up multiple matrix groups with different dimensions using vectors.
2. Initialize input matrices with random values using the `runner_vec` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Execute single GroupedGemm approach with comprehensive algorithm testing.
6. Execute multiple GroupedGemm approach with 8 separate grouped instances.
7. For each approach:
   a. Query all available algorithms using `getAllAlgos`.
   b. Create GroupedGemm objects and configure problem dimensions.
   c. Validate algorithm support and workspace requirements.
   d. Initialize with valid algorithms and execute operations.
8. Demonstrate advanced memory management with smart pointers.
9. Clean up all allocations and destroy hipBLASLt handles.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Algorithm Discovery**:
  - `hipblaslt_ext::getAllAlgos()`: Retrieves all available algorithms for a given Grouped GEMM configuration.

- **Grouped GEMM Object**:
  - `hipblaslt_ext::GroupedGemm`: A C++ class that simplifies grouped GEMM operations.
  - `setProblem()`: Configures the problems for all groups, including their dimensions, batch counts, epilogues, and inputs.
  - `isAlgoSupported()`: Checks if a specific algorithm is supported for the current problem configuration and returns its workspace size.
  - `initialize()`: Initializes the Grouped GEMM operation with a selected algorithm and workspace.
  - `run()`: Executes the Grouped GEMM operation.

- **GEMM Configuration**:
  - `hipblaslt_ext::GemmPreference`: Sets user preferences, such as the maximum workspace size with `setMaxWorkspaceBytes()`.
  - `hipblaslt_ext::GemmInputs`: Specifies the input matrices and scalars for each GEMM group.

- **User Arguments**:
  - `hipblaslt_ext::UserArguments`: A structure for passing runtime parameters to the Grouped GEMM operation.
  - `getDefaultValueForDeviceUserArguments()`: Retrieves the default user arguments for the configured problems.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblaslt_ext::GemmType`: Specifies the type of GEMM operation, such as `HIPBLASLT_GROUPED_GEMM`.

## Demonstrated API Calls

### hipBLASLt Extension API

- `hipblaslt_ext::GroupedGemm` (constructor)
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
- `hipblaslt_ext::GroupedGemm::setProblem`
- `hipblaslt_ext::GroupedGemm::getDefaultValueForDeviceUserArguments`
- `hipblaslt_ext::GroupedGemm::isAlgoSupported`
- `hipblaslt_ext::GroupedGemm::initialize`
- `hipblaslt_ext::GroupedGemm::run`
- `hipblaslt_ext::getAllAlgos`
- `hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM`

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

### Memory Management

- `HipBufferDeleter`
- `HipArrayBufferPtr<T>`
- `make_host_hip_array_buffer_ptr<T>`
- `make_device_hip_array_buffer_ptr<T>`

### Data Types and Enums

- `hipblasLtHandle_t`
- `hipblasLtMatmulHeuristicResult_t`
- `hipblasOperation_t`
- `hipblasComputeType_t`
- `hipDataType`
- `hipblasLtHalf`
- `hipblaslt_ext::UserArguments`
- `hipblaslt_ext::GemmType`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLAS_STATUS_SUCCESS`
