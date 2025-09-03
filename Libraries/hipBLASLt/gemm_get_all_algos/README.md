# hipBLASLt GEMM with Algorithm Discovery and Validation

## Description

This example illustrates the use of the `hipBLASLt` library for general matrix multiplication with comprehensive algorithm discovery and validation.

The operation calculates the following product:

$D = \alpha \cdot op_A(A) \cdot op_B(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$
- $op_A(A)$ and $op_B(B)$ are the result of applying to matrices $A$ and $B$.

## Application flow

1. Set up matrix dimensions and allocate memory for input and output matrices.
2. Initialize input matrices with random values using the `runner` utility class.
3. Copy input matrices from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Discover all available algorithms using `hipblaslt_ext::getAllAlgos()`.
6. Create matrix layout descriptors for all matrices (A, B, C, D).
7. Configure batch processing parameters if batch count > 1.
8. Create matrix multiplication descriptor and set operation attributes.
9. Set up matrix multiplication preferences including workspace size.
10. Validate each discovered algorithm using `hipblaslt_ext::matmulIsAlgoSupported()`.
11. Filter algorithms based on workspace size constraints.
12. Select the first valid algorithm and perform matrix multiplication.
13. Copy the result matrix from device to host memory.
14. Clean up hipBLASLt descriptors and device allocations.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Algorithm Discovery**:
  - `hipblaslt_ext::getAllAlgos()`: Retrieves all available algorithms for a given GEMM configuration.
  - `hipblaslt_ext::matmulIsAlgoSupported()`: Checks if a specific algorithm is supported for the current problem configuration and returns its workspace size.

- **Matrix Descriptors**:
  - `hipblasLtMatrixLayoutCreate()`: Creates a descriptor for a matrix, defining its data type, dimensions, and leading dimension.
  - `hipblasLtMatrixLayoutDestroy()`: Frees the matrix layout descriptor.

- **GEMM Operation Descriptor**:
  - `hipblasLtMatmulDescCreate()`: Creates a descriptor for the GEMM operation, specifying the computation precision.
  - `hipblasLtMatmulDescSetAttribute()`: Sets details of the GEMM operation.
  - `hipblasLtMatmulDescDestroy()`: Frees the GEMM operation descriptor.

- **Algorithm Selection**:
  - `hipblasLtMatmulPreferenceCreate()`: Creates a preference object to guide the algorithm selection process.
  - `hipblasLtMatmulPreferenceSetAttribute()`: Specifies user preferences, like the maximum workspace memory (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`).
  - `hipblasLtMatmulPreferenceDestroy()`: Frees the preference object.

- **Execution**:
  - `hipblasLtMatmul()`: Executes the matrix multiplication using a selected algorithm.

- **Key Enumerations**:
  - `hipblasOperation_t`: Specifies matrix transformations, such as `HIPBLAS_OP_N` (no transpose).
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision).
  - `hipblasComputeType_t`: Sets the precision for the computation (e.g., `HIPBLAS_COMPUTE_32F`).
  - `hipblaslt_ext::GemmType`: Specifies the type of GEMM operation, such as `HIPBLASLT_GEMM`.

## Demonstrated API Calls

### hipBLASLt Extension API

- `hipblaslt_ext::getAllAlgos`
- `hipblaslt_ext::matmulIsAlgoSupported`
- `hipblaslt_ext::GemmType::HIPBLASLT_GEMM`

### hipBLASLt Core API

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
- `hipblasLtHalf`
- `hipblaslt_ext::GemmType`
- `HIPBLAS_OP_N`
- `HIPBLAS_COMPUTE_32F`
- `HIP_R_16F`
- `HIP_R_32F`
- `HIPBLASLT_EPILOGUE_DEFAULT`
