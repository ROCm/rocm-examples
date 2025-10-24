# hipSPARSE Sparse-Dense Vector Addition (AXPYI)

## Description

This example illustrates the use of the `hipSPARSE` library for sparse-dense vector addition.

The operation calculates the following product:

$y = y + \alpha \cdot x$

where

- $\alpha$ is a scalar
- $x$ is a sparse vector with specified non-zero elements
- $y$ is a dense vector

The sparse vector $x$ is represented by:

- An array of non-zero values `x_val`
- An array of corresponding indices `x_ind`
- The number of non-zero elements `nnz`

## Application flow

1. Set up sparse vector parameters including number of non-zeros and initialize sparse index and value arrays.
2. Initialize dense vector with sequential values.
3. Define scalar multiplier alpha and index base (zero-based indexing).
4. Allocate device memory for sparse indices, sparse values, and dense vector.
5. Copy input data from host to device memory using `hipMemcpy`.
6. Set up hipSPARSE handle with `hipsparseCreate()`.
7. Perform sparse-dense vector addition using `hipsparseDaxpyi()`.
8. Copy result vector from device to host memory.
9. Clean up hipSPARSE handle and device memory allocations.

## Key APIs and Concepts

- **hipSPARSE Initialization**: The hipSPARSE library is initialized by creating a handle with `hipsparseCreate()` and released with `hipsparseDestroy()`.

- **Sparse Vector Representation**:
  - Sparse vectors are represented using coordinate (COO) format with separate arrays for values and indices.
  - The `hipsparseIndexBase_t` enum specifies whether indices are zero-based (`HIPSPARSE_INDEX_BASE_ZERO`) or one-based (`HIPSPARSE_INDEX_BASE_ONE`).

- **Sparse-Dense Operations**:
  - `hipsparseDaxpyi()`: Performs the operation y = y + α*x where x is sparse and y is dense.
  - The 'D' prefix indicates double-precision floating-point operations.
  - The 'I' suffix indicates that the sparse vector uses index arrays.

- **Memory Management**:
  - Device memory allocation and deallocation using HIP runtime functions.
  - Asynchronous and synchronous memory transfers between host and device.

## Demonstrated API Calls

### hipSPARSE

- `hipsparseDaxpyi`
- `hipsparseCreate`
- `hipsparseDestroy`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`

### Data Types and Enums

- `hipsparseHandle_t`
- `hipsparseIndexBase_t`
- `HIPSPARSE_INDEX_BASE_ZERO`
