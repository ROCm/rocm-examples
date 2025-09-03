# hipBLASLt GEMM with Matrix A Swizzling and Vector Scaling

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication combining matrix A swizzling optimization with vector-based input matrix scaling.

The operation calculates the following scaled and swizzled matrix product:

$D = \alpha \cdot (ScaleA_{vec} \circ op_A(A_{swizzled})) \cdot (ScaleB_{vec} \circ op_B(B)) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $ScaleA_{vec}$ is a vector of scaling factors with length $m$ (applied column-wise to matrix A)
- $ScaleB_{vec}$ is a vector of scaling factors with length $n$ (applied column-wise to matrix B)
- $\circ$ denotes element-wise multiplication (broadcast scaling)
- $A_{swizzled}$ is matrix A with optimized memory layout (swizzled pattern)
- $A$ is a matrix of dimensions $k \times m$ (with transpose operation)
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Create scaling factor vectors for matrices A and B with uniform values.
3. Execute two GEMM configurations for comparison:
   a. **Non-swizzled GEMM**: Matrix multiplication with vector scaling but without swizzling
   b. **Swizzled GEMM**: Matrix multiplication with both vector scaling and matrix A swizzling
4. For each configuration:
   a. Initialize input matrices with random 8-bit floating point values
   b. Copy input matrices and scaling vectors from host to device memory
   c. Apply swizzling transformation to matrix A when enabled
   d. Configure vector scaling mode and scaling factor pointers
   e. Execute matrix multiplication with integrated scaling
   f. Copy results back to host memory
5. Validate results by comparing swizzled and non-swizzled outputs.
6. Display validation results and completion status.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutSetAttribute()`: Configures matrix properties. For swizzling, `HIPBLASLT_MATRIX_LAYOUT_ORDER` is set to `HIPBLASLT_ORDER_COL16_4R16`.
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation, including matrix transformations and scaling modes. `HIPBLASLT_MATMUL_DESC_A_SCALE_MODE` and `HIPBLASLT_MATMUL_DESC_B_SCALE_MODE` are used to enable vector scaling, and the scaling vectors are provided via `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER` and `HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER`.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the matrix multiplication with swizzling and vector scaling using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_T` (transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_8F_E4M3_FNUZ`, `HIP_R_16BF`).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtOrder_t`: Defines the memory layout of a matrix, such as `HIPBLASLT_ORDER_COL16_4R16` for swizzling.
  - `hipblasLtMatmulMatrixScale_t`: Defines the scaling mode for a matrix, such as `HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`.

## Demonstrated API Calls

### hipBLASLt

- `hipblasLtMatmul`
- `hipblasLtMatrixLayoutCreate`
- `hipblasLtMatrixLayoutSetAttribute`
- `hipblasLtMatrixLayoutDestroy`
- `hipblasLtMatmulDescCreate`
- `hipblasLtMatmulDescSetAttribute`
- `hipblasLtMatmulDescDestroy`
- `hipblasLtMatmulPreferenceCreate`
- `hipblasLtMatmulPreferenceSetAttribute`
- `hipblasLtMatmulPreferenceDestroy`
- `hipblasLtMatmulAlgoGetHeuristic`
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
- `hipMemcpyDeviceToDevice`
- `hipMemcpyHostToDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`

### Data Types and Enums

- `hipblasLtHandle_t`
- `hipblasLtMatrixLayout_t`
- `hipblasLtMatmulDesc_t`
- `hipblasLtMatmulPreference_t`
- `hipblasLtMatmulHeuristicResult_t`
- `hipblasLtOrder_t`
- `hipblasLtMatmulMatrixScale_t`
- `hipblasOperation_t`
- `hipblasComputeType_t`
- `hipDataType`
- `hipblaslt_f8_fnuz`
- `hip_bfloat16`
- `HIPBLAS_OP_T`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_8F_E4M3_FNUZ`
- `HIP_R_16BF`
- `HIP_R_32F`
- `HIPBLASLT_ORDER_COL16_4R16`
- `HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`
- `HIPBLASLT_MATRIX_LAYOUT_ORDER`
- `HIPBLASLT_MATMUL_DESC_TRANSA`
- `HIPBLASLT_MATMUL_DESC_TRANSB`
- `HIPBLASLT_MATMUL_DESC_A_SCALE_MODE`
- `HIPBLASLT_MATMUL_DESC_B_SCALE_MODE`
- `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`
- `HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER`
- `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`
