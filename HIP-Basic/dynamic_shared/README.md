# HIP-Basic Dynamic Shared Memory Example

## Description

This program showcases an implementation of a simple matrix tranpose kernel, which uses shared memory that is dynamically allocated at runtime.

### Application flow

1. A number of constants are defined to control the problem details and the kernel launch parameters.
2. Input matrix is set up in host memory.
3. The necessary amount of device memory is allocated and input is copied to the device.
4. The required amount of shared memory is calculated.
5. The GPU transposition kernel is launched with previously defined arguments. The required amount of shared memory is specified.
6. The transposed matrix is copied back to the host and all device memory is freed.
7. The elements of the result matrix are compared with the expected result. The result of the comparison is printed to the standard output.

## Key APIs and Concepts

Global memory is the main memory on a GPU. This memory is used when transferring data between host and device. It has a large capacity, but also has a relatively high latency to access, which limits the performance of parallel programs. To help mitigate the effects of global memory latency, each GPU multiprocessor is equipped with a local amount of _shared_ memory. Shared memory is accessible by all threads in a multiprocessor and is typically much faster than using global memory. Each multiprocessor on a GPU has a fixed amount of shared memory, typically between 32 and 64 kilobytes. In HIP code, variables can be declared to be placed in shared memory by using the `__shared__` attribute.

A GPU multiprocessor can process multiple blocks of a kernel invocation simultaneously. In order to allocate shared memory for each block, the GPU runtime needs to know the total shared memory that each kernel can use, so that it can calculate how many groups can run at the same time. When declaring shared variables of which the size is known at compile time, the compiler computes the total size automatically. Some times, however, this size may not be known in advance, for example when the required amount of shared memory depends on the input size. In these cases, it is not beneficial to declare an upper bound, as this may unnecessarily limit the number of blocks that can be processed at the same time. In these situations _dynamic shared memory_ can be used. This is an amount of shared memory of which the size may be given at runtime. Dynamic shared memory is used by declaring an `extern` shared variable of a variable-length array of unspecified size:

```c++
extern __shared__ type var[];
```

The GPU runtime still needs to know the total amount of shared memory that a kernel will use, and for this reason this value needs to be passed with the execution configuration when launching the kernel. When using the `myKernelName<<<...>>>` kernel launch syntax, this is simply a parameter that indicates the required amount:

```c++
kernel_name<<<grid_dim, block_dim, dynamic_shared_memory_size, stream>>>(<kernel arguments>);
```

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`
- `__shared__`
- `__syncthreads`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
