# hipThreads Sparse Matrix Multiplication Step 1: CPU Baseline Example

## Description

This example implements sparse matrix-matrix multiplication (SpMM) on the CPU: given two sparse matrices A (in CSR format) and B (in CSC format) read from MatrixMarket files, it computes C = A × B. The MatrixMarket format is first loaded as a coordinate (COO) matrix, then converted to CSR or CSC as needed.

This is the single-threaded CPU baseline for the SpMM porting series. Steps 2 through 3 add parallelism and progressively move the computation to the GPU using hipThreads.

Sample matrices are provided in the `data/` directory.

### Application flow

1. Read matrix A from the first MatrixMarket file (`.mtx`) into a COO representation.
2. Convert COO to CSR format.
3. Read matrix B from the second MatrixMarket file into COO, then convert to CSC format.
4. Compute C = A × B row by row: for each row of A, iterate over the non-zero entries and accumulate contributions from the corresponding columns of B.
5. Print a summary of the result matrix (number of non-zeros, elapsed time).
