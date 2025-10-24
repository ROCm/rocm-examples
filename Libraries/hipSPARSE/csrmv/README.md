# hipSPARSE CSR Matrix-Vector Multiplication (CSRMV)

## Description

This example illustrates the use of the `hipSPARSE` library for Compressed Sparse Row (CSR) matrix-vector multiplication.

The operation calculates the following product:

$y = \alpha \cdot A \cdot x + \beta \cdot y$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a sparse matrix in CSR format
- $x$ is a dense vector
- $y$ is a dense vector

The CSR matrix $A$ is represented by:

- `val`: Array of non-zero values
- `col_ind`: Array of column indices for each non-zero value
- `row_ptr`: Array of row pointers indicating the start of each row in `val` and `col_ind`

## Application flow

1. Parse command-line arguments for matrix dimensions, trials, and batch size.
2. Set up hipSPARSE handle and query device properties.
3. Generate a 2D Laplacian matrix in CSR format using the utility function `gen_2d_laplacian()`.
4. Initialize random scalar values alpha and beta, and dense vector x.
5. Create matrix descriptor using `hipsparseCreateMatDescr()`.
6. Allocate device memory for CSR matrix data (row_ptr, col_ind, val) and vectors (x, y).
7. Copy input data from host to device memory.
8. Perform warm-up iterations to ensure optimal performance.
9. Execute CSR matrix-vector multiplication for specified number of trials using `hipsparseDcsrmv()`.
10. Measure execution time and calculate performance metrics (GFLOPS, bandwidth).
11. Clean up matrix descriptor, hipSPARSE handle, and device memory allocations.

## Key APIs and Concepts

- **hipSPARSE Handle Management**: The hipSPARSE library context is managed through a handle created with `hipsparseCreate()` and released with `hipsparseDestroy()`.

- **Matrix Descriptors**:
  - `hipsparseCreateMatDescr()`: Creates a descriptor that contains matrix properties like index base and storage format.
  - `hipsparseDestroyMatDescr()`: Releases the matrix descriptor resources.
  - Matrix descriptors are required for most sparse matrix operations in hipSPARSE.

- **CSR Matrix Operations**:
  - `hipsparseDcsrmv()`: Performs CSR matrix-vector multiplication with double precision.
  - The 'D' prefix indicates double-precision floating-point operations.
  - Supports different operation types via `hipsparseOperation_t` (transpose, non-transpose).

- **Performance Measurement**:
  - Uses custom timing functions with device synchronization for accurate performance measurement.
  - Calculates theoretical GFLOPS based on the number of arithmetic operations (2 * nnz for SpMV).
  - Computes memory bandwidth based on data transferred during the operation.

- **2D Laplacian Generation**:
  - The `gen_2d_laplacian()` utility generates a sparse matrix representing a 2D discrete Laplacian operator.
  - Creates a 5-point stencil pattern commonly used in scientific computing applications.

## Demonstrated API Calls

### hipSPARSE

- `hipsparseCreateMatDescr`
- `hipsparseDestroyMatDescr`
- `hipsparseDcsrmv`
- `hipsparseCreate`
- `hipsparseDestroy`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipGetDevice`
- `hipGetDeviceProperties`
- `hipMalloc`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`

### Data Types and Enums

- `hipsparseHandle_t`
- `hipsparseMatDescr_t`
- `hipsparseOperation_t`
- `hipDeviceProp_t`
- `HIPSPARSE_OPERATION_NON_TRANSPOSE`
