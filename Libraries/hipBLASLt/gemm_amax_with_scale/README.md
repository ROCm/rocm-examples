# hipBLASLt GEMM with Absolute Maximum and Output Scaling

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication with integrated absolute maximum computation and output scaling.

The operation calculates the following product with AMAX computation and scaling:

$D = scale \cdot (\alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C)$

$AMAX = \max(|D_{i,j}|)$

where

- $\alpha$ and $\beta$ are scalars
- $scale$ is an input scaling factor (set to 0.5 in this example)
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $AMAX$ is the absolute maximum value of all elements in the scaled output matrix $D$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Allocate additional memory for AMAX output and scaling factor.
4. Initialize scaling factor (0.5) and copy to device memory.
5. Copy input matrices from host to device memory.
6. Set up hipBLASLt handle and stream.
7. Create matrix layout descriptors for all matrices (A, B, C, D).
8. Create matrix multiplication descriptor and set operation attributes.
9. Configure AMAX and scaling attributes in the matrix multiplication descriptor.
10. Set up matrix multiplication preferences including workspace size.
11. Query heuristic algorithms to find optimal implementation.
12. Perform matrix multiplication with integrated AMAX computation and output scaling.
13. Handle error cases with proper cleanup if no valid solutions are found.
14. Clean up AMAX-related allocations and hipBLASLt descriptors.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation. For AMAX computation, `HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER` is used to specify the output pointer for the absolute maximum value. For output scaling, `HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER` is used to provide a scaling factor.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulAlgoGetHeuristic()`: Queries for a list of high-performance algorithms based on the operation descriptors and preferences.
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the matrix multiplication, including the AMAX computation and output scaling, using a selected algorithm from the heuristic results.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_8F_E4M3_FNUZ` for 8-bit floating point).
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
- `hipblasLtEpilogue_t`
- `hipDataType`
- `hipblaslt_f8_fnuz`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_8F_E4M3_FNUZ`
- `HIP_R_32F`
- `HIPBLASLT_EPILOGUE_DEFAULT`
- `HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER`
- `HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER`
