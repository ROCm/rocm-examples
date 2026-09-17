# hipThreads Sparse Matrix Multiplication Step 3: hipThreads GPU Port Example

## Description

This example completes the GPU port of SpMM using hipThreads. The multiplication kernel runs on the GPU via `hip::wthread`, with matrix data held in GPU-resident memory. The pre-processing steps (MatrixMarket loading and sorting) remain on the CPU.

Three versions of the matrix struct are used: a host version with `std::unique_ptr`, a device version with `thrust::unique_ptr`, and a raw-pointer version for passing into `__device__` lambdas (since GPU lambdas cannot capture smart pointers).

### Application flow

1. Read matrices from MatrixMarket files and sort rows on the CPU.
2. Construct `CSRMatrix_d` and `CSCMatrix_d` by copying host data to GPU-resident arrays.
3. Convert to raw-pointer `CSRMatrix_raw` / `CSCMatrix_raw` structs for use inside `__device__` lambdas.
4. Spawn `hip::wthread::hardware_concurrency()` GPU threads, each processing a strided subset of output rows of C = A × B.
5. Join all threads, copy the result matrix back to the host, and print the summary and elapsed time.

## Key APIs and Concepts

### hipThreads

- `hip::wthread` — GPU thread that executes a `__device__` lambda. Each thread processes a strided range of output rows.

- `hip::wthread::hardware_concurrency()` — returns the number of available GPU wavefronts, analogous to `std::thread::hardware_concurrency()`.

### Data structure pattern

Because `__device__` lambdas cannot capture or use smart pointers, a three-tier struct pattern is used:

- `CSRMatrix` / `CSCMatrix` — host structs with `std::unique_ptr` members.

- `CSRMatrix_d` / `CSCMatrix_d` — device structs with `thrust::unique_ptr` members. Constructed by copying host data.

- `CSRMatrix_raw` / `CSCMatrix_raw` — plain raw-pointer structs produced by the device structs. These are what the `__device__` lambda receives and operates on.

## Demonstrated API Calls

### hipThreads

- `hip::wthread`

- `hip::wthread::hardware_concurrency`

- `hip::wthread::join`

### rocThrust

- `thrust::unique_ptr`

- `thrust::copy`
