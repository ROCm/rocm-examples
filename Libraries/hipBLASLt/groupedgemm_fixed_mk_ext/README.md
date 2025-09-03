# hipBLASLt Extension API - Grouped GEMM with Fixed M,K and Variable N

## Description

This example illustrates the use of the `hipBLASLt` extension API for grouped general matrix multiplication with fixed M and K dimensions but variable N dimensions.

The operation calculates multiple matrix products where M and K are fixed but N varies per group:

$D_i = \alpha_i \cdot op_A(A_i) \cdot op_B(B_i) + \beta_i \cdot C_i$ for $i = 0, 1, ..., N-1$

where

- $\alpha_i$ and $\beta_i$ are scalars for each group
- $A_i$ is a matrix of dimensions $m \times k$ (fixed for all groups)
- $B_i$ is a matrix of dimensions $k \times n_i$ (variable N dimension)
- $C_i$ and $D_i$ are matrices of dimensions $m \times n_i$
- M and K dimensions are constant across all groups
- N dimension varies per group and can be updated dynamically
- $op_A(A_i)$ and $op_B(B_i)$ are the result of applying to matrices $A_i$ and $B_i$.

## Application flow

1. Set up multiple matrix groups with fixed M,K but variable N dimensions.
2. Initialize input matrices with random values using the `runner_vec` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create GEMM preference object and set workspace size.
6. Create GroupedGemm object with mixed precision data types.
7. Calculate sum of all N dimensions for workspace allocation strategy.
8. Copy variable N dimensions to device memory.
9. Set problem dimensions using sum-of-N approach for efficient memory usage.
10. Get default user arguments and copy to device memory.
11. Query all available algorithms and validate workspace requirements.
12. For each valid algorithm, initialize and execute multiple iterations.
13. Use GPU kernel to dynamically update N dimensions in user arguments.
14. Execute grouped GEMM operations with updated parameters.
15. Clean up device allocations including N dimension arrays.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Grouped GEMM Object**:
  - `hipblaslt_ext::GroupedGemm`: A C++ class that simplifies grouped GEMM operations with fixed M and K dimensions.
  - `setProblem()`: Configures the problems for all groups, using a sum-of-N strategy for efficient workspace allocation.
  - `algoGetHeuristic()`: Queries for a list of high-performance algorithms.
  - `initialize()`: Initializes the Grouped GEMM operation with a selected algorithm and workspace.
  - `run()`: Executes the Grouped GEMM operation.

- **GEMM Configuration**:
  - `hipblaslt_ext::GemmPreference`: Sets user preferences, such as the maximum workspace size with `setMaxWorkspaceBytes()`.
  - `hipblaslt_ext::GemmInputs`: Specifies the input matrices and scalars for each GEMM group.

- **Dynamic N Dimension Update**:
  - A custom HIP kernel (`kernel_update_n`) is used to dynamically update the N dimension for each group in the `hipblaslt_ext::UserArguments` structure on the device.

- **User Arguments**:
  - `hipblaslt_ext::UserArguments`: A structure for passing runtime parameters to the Grouped GEMM operation.
  - `getDefaultValueForDeviceUserArguments()`: Retrieves the default user arguments for the configured problems.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_8F_E4M3_FNUZ`, `HIP_R_16F`).
  - `hipblasComputeType_t`: Sets the precision for the computation, such as `HIPBLAS_COMPUTE_32F_FAST_16F`.

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
- `hipLaunchKernelGGL`

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
- `HIPBLAS_COMPUTE_32F_FAST_16F`
- `HIP_R_8F_E4M3_FNUZ`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLAS_STATUS_SUCCESS`
