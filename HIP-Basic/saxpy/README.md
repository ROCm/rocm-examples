# HIP-Basic "SAXPY" Example

## Description

This program demonstrates a simple implementation of the "SAXPY" kernel. The "S" stands for single-precision (i.e. `float`) and "AXPY" stands for the operation performed: $Y_i=aX_i+Y_i$. The simple nature of this example makes it an ideal starting point for developers who are just getting introduced to HIP.

### Application flow

1. A number of constants are defined to control the problem details and the kernel launch parameters.
2. The two input vectors, $X$ and $Y$ are instantiated in host memory. $X$ is filled with an incrementing sequence starting from 1, whereas $Y$ is filled with ones.
3. The necessary amount of device (GPU) memory is allocated and the elements of the input vectors are copied to the device memory.
4. A trace message is printed to the standard output.
5. The GPU kernel is launched with the previously defined arguments.
6. The results are copied back to host vector $Y$.
7. The previously allocated device memory is freed.
8. The first few elements of the result vector are printed to the standard output.

## Key APIs and Concepts

- `hipMalloc` is used to allocate memory in the global memory of the device (GPU). This is usually necessary, since the kernels running on the device cannot access host (CPU) memory (unless it is device-accessible pinned host memory, see `hipHostMalloc`). Beware, that the memory returned is uninitialized.
- `hipFree` de-allocates device memory allocated by `hipMalloc`. It is necessary to free no longer used memory with this function to avoid resource leakage.
- `hipMemcpy` is used to transfer bytes between the host and the device memory in both directions. A call to it synchronizes the device with the host, meaning that all kernels queued before `hipMemcpy` will finish before the copying starts. The function returns once the copying has finished.
- `myKernelName<<<gridDim, blockDim, dynamicShared, stream>>>(kernelArguments)` queues the execution of the provided kernel on the device. It is asynchronous, the call may return before the execution of the kernel is finished. Its arguments come as the following:
  - The kernel (`__global__`) function to launch.
  - The number of blocks in the kernel grid, i.e. the grid size. It can be up to 3 dimensions.
  - The number of threads in each block, i.e. the block size. It can be up to 3 dimensions.
  - The amount of dynamic shared memory provided for the kernel, in bytes. Not used in this example.
  - The device stream, on which the kernel is queued. In this example, the default stream is used.
  - All further arguments are passed to the kernel function. Notice, that built-in and simple (POD) types may be passed to the kernel, but complex ones (e.g. `std::vector`) usually cannot be.
- `hipGetLastError` returns the error code resulting from the previous operation.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`

#### Host symbols

- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
- `hipGetLastError`
