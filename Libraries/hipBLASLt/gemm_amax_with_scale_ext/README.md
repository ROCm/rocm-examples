# hipBLASLt Extension API - GEMM with Absolute Maximum and Output Scaling

## Description

This example illustrates the use of the `hipBLASLt` extension API for general matrix multiplication with integrated absolute maximum computation and output scaling.

The operation calculates the following product with AMAX computation and scaling:

$D = scale \cdot (\alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C)$

$AMAX = \max(|D_{i,j}|)$

where

- $\alpha$ and $\beta$ are scalars
- $scale$ is an output scaling factor (set to 2.0 in this example)
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $AMAX$ is the absolute maximum value of all elements in the scaled output matrix $D$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create GEMM preference object and set workspace size.
6. Create GEMM object with data types and compute type.
7. Allocate device memory for AMAX output value and scaling factor.
8. Initialize scaling factor (2.0) and copy to device memory.
9. Set up GEMM epilogue (default operation).
10. Configure GEMM inputs including matrices, AMAX output pointer, and scaling factor.
11. Set problem dimensions and configure the GEMM operation.
12. Query heuristic algorithms to find optimal implementation.
13. Initialize GEMM with selected algorithm and workspace.
14. Execute the GEMM operation with integrated AMAX computation and output scaling.
15. Clean up device allocations and destroy hipBLASLt handle.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **GEMM Object**:
  - `hipblaslt_ext::Gemm`: A C++ class that simplifies GEMM operations.
  - `setProblem()`: Configures the GEMM problem dimensions, epilogue, and inputs.
  - `algoGetHeuristic()`: Queries for a list of high-performance algorithms.
  - `initialize()`: Initializes the GEMM operation with a selected algorithm and workspace.
  - `run()`: Executes the GEMM operation.

- **GEMM Configuration**:
  - `hipblaslt_ext::GemmPreference`: Sets user preferences, such as the maximum workspace size with `setMaxWorkspaceBytes()`.
  - `hipblaslt_ext::GemmInputs`: Specifies the input matrices and scalars for the GEMM operation. `setAmaxD()` is used to enable AMAX computation on the output matrix, and `setScaleD()` is used to provide an output scaling factor.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_8F_E4M3_FNUZ` for 8-bit floating point).
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
- `hipblaslt_ext::GemmInputs::setAmaxD`
- `hipblaslt_ext::GemmInputs::setScaleD`
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
- `hipDataType`
- `hipblaslt_f8_fnuz`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_8F_E4M3_FNUZ`
- `HIP_R_32F`
