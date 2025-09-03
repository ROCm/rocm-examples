# hipBLASLt Extension API - Mixed Precision GEMM with Validation

## Description

This example illustrates the use of the `hipBLASLt` extension API for mixed precision general matrix multiplication with matrix scaling and result validation.

The operation calculates the following product with mixed precision and scaling:

$D = \alpha \cdot (scale_A \cdot op_A(A)) \cdot op_B(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $scale_A$ is a scaling factor applied to matrix A (set to 2.0 in this example)
- $A$ is a matrix of dimensions $m \times k$ stored in 8-bit floating point (E4M3 FNUZ)
- $B$ is a matrix of dimensions $k \times n$ stored in 16-bit floating point
- $C$ and $D$ are matrices of dimensions $m \times n$ stored in 32-bit floating point
- Computation is performed using 32-bit floating point with 16-bit fast mode
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices with different precisions.
2. Initialize input matrices with random values using the `runner` utility class with mixed types.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create GEMM preference object and set workspace size.
6. Create GEMM object with mixed data types and fast 16-bit compute type.
7. Allocate and configure scaling factor for matrix A.
8. Set up GEMM epilogue (default operation).
9. Configure GEMM inputs including matrices and scaling factor.
10. Set problem dimensions and configure the GEMM operation.
11. Query heuristic algorithms to find optimal implementation.
12. Initialize GEMM with selected algorithm and workspace.
13. Execute mixed precision GEMM operation with scaling.
14. Validate results using CPU reference implementation.
15. Clean up scaling allocation and device resources.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **GEMM Object**:
  - `hipblaslt_ext::Gemm`: A C++ class that simplifies mixed-precision GEMM operations.
  - `setProblem()`: Configures the GEMM problem dimensions, batch count, epilogue, and inputs.
  - `algoGetHeuristic()`: Queries for a list of high-performance algorithms.
  - `initialize()`: Initializes the GEMM operation with a selected algorithm and workspace.
  - `run()`: Executes the GEMM operation.

- **GEMM Configuration**:
  - `hipblaslt_ext::GemmPreference`: Sets user preferences, such as the maximum workspace size with `setMaxWorkspaceBytes()`.
  - `hipblaslt_ext::GemmInputs`: Specifies the input matrices, scalars, and scaling factors for the GEMM operation. `setScaleA()` is used to set the scaling factor for matrix A.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_8F_E4M3_FNUZ`, `HIP_R_16F`, `HIP_R_32F`).
  - `hipblasComputeType_t`: Sets the precision for the computation, such as `HIPBLAS_COMPUTE_32F_FAST_16F`.

## Demonstrated API Calls

### hipBLASLt Extension API

- `hipblaslt_ext::Gemm` (constructor with mixed types)
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
- `hipblaslt_ext::GemmInputs::setScaleA`
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
- `hipMemcpyDtoH`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`

### Data Types and Enums

- `hipblasLtHandle_t`
- `hipblasLtMatmulHeuristicResult_t`
- `hipblasOperation_t`
- `hipblasComputeType_t`
- `hipDataType`
- `hipblaslt_f8_fnuz`
- `hipblasLtHalf`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F_FAST_16F`
- `HIP_R_8F_E4M3_FNUZ`
- `HIP_R_16F`
- `HIP_R_32F`
