# HIP-Doc Matrix Multiplication Example

## Description

This example demonstrates simple matrix multiplication using HIP, comparing CPU
and GPU implementations. The program computes the matrix product C = A × B
using both a CPU reference implementation and a GPU kernel, then validates that
both produce matching results.

For more information on HIP programming, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/).

### Application flow

1. Two input matrices (A and B) and two output matrices (one for CPU, one for GPU) are allocated on the host.
2. The matrices are randomly initialized with floating-point values.
3. The CPU matrix multiplication is performed as a reference implementation.
4. Device memory is allocated for matrices A, B, and the output matrix C.
5. The input matrices are copied from host to device memory.
6. The GPU matrix multiplication kernel is launched with proper grid and block dimensions.
7. The kernel launch is checked for errors and the device is synchronized.
8. The GPU result matrix is copied back from device to host memory.
9. The CPU and GPU results are compared element-by-element to verify they match.
10. All device and host memory is freed.

### Matrix Multiplication Implementation

The GPU implementation uses a straightforward approach where each thread computes a single element of the output matrix by:

- Computing its global row and column indices using block and thread indices
- Performing a dot product between the corresponding row of matrix A and column of matrix B
- Writing the result to the output matrix

## Key APIs and Concepts

### HIP Runtime APIs

- `hipMalloc`: Allocates device memory
- `hipMemcpy`: Transfers data between host and device
- `hipFree`: Frees device memory
- `hipGetLastError`: Retrieves the last error from a runtime call
- `hipDeviceSynchronize`: Blocks until all device operations complete
- `hipLaunchKernelGGL`: Launches a kernel function on the GPU

### Device Code Features

- `__global__`: Declares a kernel function callable from host
- `blockIdx`, `blockDim`, `threadIdx`: Built-in variables for grid/block indexing

## Configuration

- Matrix dimensions: 32 × 32
- Block size: 16 × 16 threads per block
- Grid size: 2 × 2 blocks

## Demonstrated API calls

### HIP runtime

#### Device symbols

- `blockDim`
- `blockIdx`
- `threadIdx`

#### Host symbols

- `hipDeviceSynchronize`
- `hipFree`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
- `hipLaunchKernelGGL`
