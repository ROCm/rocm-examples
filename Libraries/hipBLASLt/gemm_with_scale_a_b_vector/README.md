# hipBLASLt GEMM with Vector Input Matrix Scaling

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication with vector-based input matrix scaling.

The operation calculates the following vector-scaled matrix product:

$D = \alpha \cdot (ScaleA_{vec} \circ op_A(A)) \cdot (ScaleB_{vec} \circ op_B(B)) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $ScaleA_{vec}$ is a vector of scaling factors with length $m$ (applied column-wise to matrix A)
- $ScaleB_{vec}$ is a vector of scaling factors with length $n$ (applied column-wise to matrix B)
- $\circ$ denotes element-wise multiplication (broadcast scaling)
- $A$ is a matrix of dimensions $k \times m$ (with transpose operation)
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Create scaling factor vectors for matrices A and B (uniform values 0.5 and 2.0 respectively).
4. Copy input matrices from host to device memory.
5. Set up hipBLASLt handle and stream.
6. Allocate device memory for scaling factor vectors and copy values from host.
7. Create matrix layout descriptors for all matrices (A, B, C, D).
8. Create matrix multiplication descriptor and set operation attributes.
9. Configure vector scaling mode for both input matrices.
10. Set device pointers to scaling factor vectors in the multiplication descriptor.
11. Set up matrix multiplication preferences including workspace size.
12. Query heuristic algorithms to find optimal implementation.
13. Perform vector-scaled matrix multiplication using the selected algorithm.
14. Copy result matrix from device to host memory.
15. Clean up scaling vector allocations and hipBLASLt descriptors.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
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
  - `hipblasLtMatmul()`: Executes the matrix multiplication with vector scaling using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_T` (transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_8F_E4M3_FNUZ`, `HIP_R_16BF`).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblasLtMatmulMatrixScale_t`: Defines the scaling mode for a matrix, such as `HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`.

## Demonstrated API Calls

### hipBLASLt

- `hipblasLtMatmul`
- `hipblasLtMatrixLayoutCreate`
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
- `hipblasLtMatmulPreference_t`
- `hipblasLtMatmulHeuristicResult_t`
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
- `HIPBLASLT_MATMUL_DESC_TRANSA`
- `HIPBLASLT_MATMUL_DESC_TRANSB`
- `HIPBLASLT_MATMUL_DESC_A_SCALE_MODE`
- `HIPBLASLT_MATMUL_DESC_B_SCALE_MODE`
- `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`
- `HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER`
- `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`
- `HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`
