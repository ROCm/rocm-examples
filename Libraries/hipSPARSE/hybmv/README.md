# hipSPARSE HYB Matrix-Vector Multiplication (HYBMV)

## Description

This example illustrates the use of the `hipSPARSE` library for Hybrid (HYB) format matrix-vector multiplication.

The operation calculates the following product:

$y = \alpha \cdot A \cdot x + \beta \cdot y$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a sparse matrix in HYB format
- $x$ is a dense vector
- $y$ is a dense vector

The HYB format is a hybrid storage format that combines:

- ELL format for rows with regular sparsity patterns
- COO format for rows with irregular sparsity patterns
- Automatically optimizes storage based on matrix structure

## Application flow

1. Parse command-line arguments for matrix dimensions, trials, and batch size.
2. Set up hipSPARSE handle and query device properties.
3. Generate a 2D Laplacian matrix in CSR format using the utility function `gen_2d_laplacian()`.
4. Initialize random scalar values alpha and beta, and dense vector x.
5. Create matrix descriptor using `hipsparseCreateMatDescr()`.
6. Allocate device memory for CSR matrix data (row_ptr, col_ind, val) and vectors (x, y).
7. Copy input data from host to device memory.
8. Create HYB matrix structure using `hipsparseCreateHybMat()`.
9. Convert CSR matrix to HYB format using `hipsparseDcsr2hyb()` with automatic partitioning.
10. Free original CSR matrix data (now stored in HYB format).
11. Perform warm-up iterations to ensure optimal performance.
12. Execute HYB matrix-vector multiplication for specified number of trials using `hipsparseDhybmv()`.
13. Measure execution time and calculate performance metrics (GFLOPS, bandwidth).
14. Clean up HYB matrix, matrix descriptor, hipSPARSE handle, and device memory allocations.

## Key APIs and Concepts

- **hipSPARSE Handle Management**: The hipSPARSE library context is managed through a handle created with `hipsparseCreate()` and released with `hipsparseDestroy()`.

- **HYB Matrix Format**:
  - `hipsparseCreateHybMat()`: Creates a HYB matrix structure that can store sparse data in an optimized hybrid format.
  - `hipsparseDestroyHybMat()`: Releases the HYB matrix structure and associated memory.
  - HYB format automatically partitions the matrix into ELL and COO components for optimal performance.

- **Matrix Format Conversion**:
  - `hipsparseDcsr2hyb()`: Converts a CSR matrix to HYB format with double precision.
  - The conversion process analyzes matrix structure to determine optimal ELL/COO partitioning.
  - `HIPSPARSE_HYB_PARTITION_AUTO` lets the library automatically choose the best partitioning strategy.

- **HYB Matrix Operations**:
  - `hipsparseDhybmv()`: Performs HYB matrix-vector multiplication with double precision.
  - The 'D' prefix indicates double-precision floating-point operations.
  - Supports different operation types via `hipsparseOperation_t` (transpose, non-transpose).

- **Performance Optimization**:
  - HYB format can provide better performance than CSR for matrices with varying row sparsity patterns.
  - The ELL component handles regular rows efficiently, while COO handles irregular rows.
  - Format conversion overhead is amortized over multiple matrix-vector operations.

- **Memory Management**:
  - HYB format can use more memory than CSR but provides better performance characteristics for certain matrix patterns.
  - Original CSR data can be freed after conversion to HYB format.

## Demonstrated API Calls

### hipSPARSE

- `hipsparseCreateHybMat`
- `hipsparseDestroyHybMat`
- `hipsparseDcsr2hyb`
- `hipsparseDhybmv`
- `hipsparseCreateMatDescr`
- `hipsparseDestroyMatDescr`
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
- `hipsparseHybMat_t`
- `hipsparseOperation_t`
- `hipDeviceProp_t`
- `HIPSPARSE_OPERATION_NON_TRANSPOSE`
- `HIPSPARSE_HYB_PARTITION_AUTO`
