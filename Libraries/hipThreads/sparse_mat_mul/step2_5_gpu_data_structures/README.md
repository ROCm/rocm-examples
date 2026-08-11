# hipThreads Sparse Matrix Multiplication Step 2.5: GPU Data Structures Example

## Description

This intermediate step ports the SpMM data structures to GPU-compatible forms without yet running the multiplication on the GPU. `std::unique_ptr<float[]>` is replaced with `thrust::unique_ptr` for GPU-resident arrays, and data is copied to device memory via rocThrust. The multiplication kernel itself still runs on the CPU via `std::thread`.

This step demonstrates the data structure migration pattern needed before a full GPU port: identify which arrays must be device-resident, allocate them with GPU-compatible smart pointers, and copy data host-to-device.

### Application flow

1. Read matrices from MatrixMarket files on the host.
2. Allocate GPU-resident `CSRMatrix_d` and `CSCMatrix_d` using `thrust::unique_ptr`, copying host data to device.
3. Pass device pointers into the multiplication kernel, which still runs on the CPU (accessing device memory via unified addressing or explicit copies).
4. Print the result summary and elapsed time.

## Key APIs and Concepts

### rocThrust

- `thrust::unique_ptr` — GPU-resident smart pointer for arrays allocated in device memory.
- `thrust::copy` — transfers array data from the host to device memory.

## Demonstrated API Calls

### rocThrust

- `thrust::unique_ptr`
- `thrust::copy`
