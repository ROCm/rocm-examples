# hipBLASLt Extension API - GEMM with Matrix A Swizzling and Bias Epilogue

## Description

This example illustrates the use of the `hipBLASLt` extension API for general matrix multiplication combining matrix A swizzling optimization with bias epilogue fusion.

The operation calculates the following swizzled matrix product with fused bias addition:

$D = \alpha \cdot op_A(A_{swizzled}) \cdot op_B(B) + \beta \cdot C + bias$

where

- $\alpha$ and $\beta$ are scalars
- $A_{swizzled}$ is matrix A with optimized memory layout (swizzled pattern)
- $bias$ is a vector of length $m$ added to each column of the result
- $A$ is a matrix of dimensions $k \times m$ (with transpose operation)
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Create matrix layout descriptors for all matrices (A, B, C, D).
6. Apply swizzling transformation to matrix A with optimized memory layout.
7. Create matrix multiplication descriptor with bias epilogue configuration.
8. Allocate and initialize bias vector on device memory.
9. Set bias pointer in the matrix multiplication descriptor.
10. Create extension API Gemm object with all matrix parameters.
11. Set up GEMM preference and query heuristic algorithms.
12. Initialize GEMM with selected algorithm and workspace.
13. Execute the GEMM operation with fused bias addition.
14. Clean up matrix descriptors and bias memory allocations.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutSetAttribute()`: Configures matrix properties. For swizzling, `HIPBLASLT_MATRIX_LAYOUT_ORDER` is set to `HIPBLASLT_ORDER_COL16_4R8`.
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation. For bias addition, the epilogue is set to `HIPBLASLT_EPILOGUE_BIAS`, and the bias pointer is configured using `HIPBLASLT_MATMUL_DESC_BIAS_POINTER`.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **GEMM Object**:
  - `hipblaslt_ext::Gemm`: A C++ class that simplifies GEMM operations, including those with swizzling and epilogues.
  - `algoGetHeuristic()`: Queries for a list of high-performance algorithms.
  - `initialize()`: Initializes the GEMM operation with a selected algorithm and workspace.
  - `run()`: Executes the GEMM operation.

- **GEMM Configuration**:
  - `hipblaslt_ext::GemmPreference`: Sets user preferences, such as the maximum workspace size with `setMaxWorkspaceBytes()`.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_T` (transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtEpilogue_t`: Defines the epilogue operation, such as `HIPBLASLT_EPILOGUE_BIAS`.
  - `hipblasLtOrder_t`: Defines the memory layout of a matrix, such as `HIPBLASLT_ORDER_COL16_4R8` for swizzling.

## Demonstrated API Calls

### hipBLASLt Extension API

- `hipblaslt_ext::Gemm` (constructor with descriptor)
- `hipblaslt_ext::GemmPreference`
- `hipblaslt_ext::GemmPreference::setMaxWorkspaceBytes`
- `hipblaslt_ext::Gemm::algoGetHeuristic`
- `hipblaslt_ext::Gemm::initialize`
- `hipblaslt_ext::Gemm::run`

### hipBLASLt Core API

- `hipblasLtCreate`
- `hipblasLtDestroy`
- `hipblasLtMatrixLayoutCreate`
- `hipblasLtMatrixLayoutSetAttribute`
- `hipblasLtMatrixLayoutDestroy`
- `hipblasLtMatmulDescCreate`
- `hipblasLtMatmulDescSetAttribute`
- `hipblasLtMatmulDescDestroy`

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
- `hipblasLtMatrixLayout_t`
- `hipblasLtMatmulDesc_t`
- `hipblasLtMatmulHeuristicResult_t`
- `hipblasLtOrder_t`
- `hipblasLtEpilogue_t`
- `hipblasOperation_t`
- `hipblasComputeType_t`
- `hipDataType`
- `hipblasLtHalf`
- `HIPBLAS_OP_T`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLASLT_ORDER_COL16_4R8`
- `HIPBLASLT_EPILOGUE_BIAS`
- `HIPBLASLT_MATRIX_LAYOUT_ORDER`
- `HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT`
- `HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`
- `HIPBLASLT_MATMUL_DESC_TRANSA`
- `HIPBLASLT_MATMUL_DESC_TRANSB`
- `HIPBLASLT_MATMUL_DESC_EPILOGUE`
- `HIPBLASLT_MATMUL_DESC_BIAS_POINTER`
