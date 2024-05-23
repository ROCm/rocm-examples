# HIP-Basic Shared Memory Example

## Description

The shared memory is an on-chip type of memory that is visible to all the threads within the same block, allowing them to communicate by writing and reading data from the same memory space. However, some synchronization among the threads of the block is needed to ensure that all of them have written before trying to access the data.

When using the appropriate access pattern, this memory can provide much less latency than local or global memory (nearly as much as registers), making it a much better option in certain cases. If the size of the shared memory to be used is known at compile time, it can be explicitly specified and it is then known as static shared memory.

This example implements a simple matrix transpose kernel to showcase how to use static shared memory.

### Application flow

1. A number of constants are defined for the kernel launch parameters.
2. The input and output matrices are allocated and initialized in host memory.
3. The necessary amount of device memory for the input and output matrices is allocated and the input data is copied to the device.
4. A trace message is printed to the standard output.
5. The GPU kernel is then launched with the previously defined arguments.
6. The transposed matrix is copied back to host memory.
7. All device memory is freed.
8. The expected transposed matrix is calculated with a CPU version of the transpose kernel and the transposed matrix obtained from the kernel execution is then compared with it. The result of the comparison is printed to the standard output.

## Key APIs and Concepts

- `__shared__` is a variable declaration specifier necessary to allocate shared memory from the device.
- `__syncthreads` allows to synchronize all the threads within the same block. This synchronization barrier is used to ensure that every thread in a block have finished writing in shared memory before another threads in the block try to access that data.
- `hipMalloc` allocates host device memory in global memory, and with `hipMemcpy` data bytes can be transferred from host to device (using `hipMemcpyHostToDevice`) or from device to host (using `hipMemcpyDeviceToHost`), among others.
- `myKernelName<<<...>>>` queues the execution of a kernel on a device (GPU).
- `hipGetLastError` gets the last error returned by any HIP runtime API call.
- `hipFree` deallocates device memory allocated with `hipMalloc`.

## Demonstrated API Calls

### HIP runtime

- `__global__`
- `__shared__`

#### Device symbols

- `blockDim`
- `blockIdx`
- `threadIdx`
- `__syncthreads`

#### Host symbols

- `hipFree`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
