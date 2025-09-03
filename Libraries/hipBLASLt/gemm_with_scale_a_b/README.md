# hipBLASLt GEMM with Input Matrix Scaling

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication with input matrix scaling.

The operation calculates the following scaled matrix product:

$D = \alpha \cdot (scale_A \cdot op_A(A)) \cdot (scale_B \cdot op_B(B)) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $scale_A$ and $scale_B$ are input matrix scaling factors
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Set scaling factors for input matrices A and B (0.5 and 2.0 respectively).
4. Copy input matrices from host to device memory.
5. Set up hipBLASLt handle and stream.
6. Allocate device memory for scaling factors and copy values from host.
7. Create matrix layout descriptors for all matrices (A, B, C, D).
8. Create matrix multiplication descriptor and set operation attributes.
9. Configure matrix scaling attributes with device pointers to scaling factors.
10. Set up matrix multiplication preferences including workspace size.
11. Query heuristic algorithms to find optimal implementation.
12. Perform scaled matrix multiplication using the selected algorithm.
13. Copy result matrix from device to host memory.
14. Clean up scaling factor allocations and hipBLASLt descriptors.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation, including matrix transformations and scaling factors for matrices A and B using `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER` and `HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER`.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the matrix multiplication with scaling using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_8F_E4M3_FNUZ`, `HIP_R_16F`).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).

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
- `hipblasOperation_t`
- `hipblasComputeType_t`
- `hipDataType`
- `hipblaslt_f8_fnuz`
- `hipblasLtHalf`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_8F_E4M3_FNUZ`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLASLT_MATMUL_DESC_TRANSA`
- `HIPBLASLT_MATMUL_DESC_TRANSB`
- `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`
- `HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER`
- `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`
