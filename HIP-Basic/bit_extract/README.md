# HIP-Basic Bit Extract Example

## Description

A HIP-specific bit extract solution is presented in this example.

### Application flow

1. Allocate memory for host vectors.
2. Fill the input host vector as an arithmetic sequence by the vector index.
3. Allocate memory for device arrays.
4. Copy the arithmetic sequence from the host to device memory.
5. Apply bit extract operator on the sequence element by element and return with result array. If we use HIP, __bitextract_u32() device function is used, otherwise the standard bit shift operator.
6. Copy the result sequence from the device to the host memory
7. Compare the result sequence to the expected sequence, element by element. If a mismatch is detected, the vector index and both values are printed, and the program exits with an error code.
8. Deallocate device and host memory.
9. "PASSED!" is printed when the flow was successful.

## Key APIs and Concepts

- `kernel_name<<<kernel_name, grid_dim, block_dim, dynamic_shared_memory_size, stream>>>(<kernel arguments>)` is the HIP kernel launcher where the grid and block dimension, dynamic shared memory size and HIP stream is defined. We use NULL stream in the recent example.
- `__bitextract_u32(source, bit_start, num_bits)` is the built-in AMD HIP bit extract operator, where we define a source scalar, a `bit_start` start bit and a `num_bits` number of extraction bits. The operator returns with a scalar value.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- `__bitextract_u32`

#### Host symbols

- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
