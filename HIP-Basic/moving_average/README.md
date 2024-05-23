# HIP-Basic Moving Average Example

## Description

This example shows the use of a kernel that computes a moving average on one-dimensional data. In a sequential program, the moving average of a given input array is found by processing the elements one by one. The average of the previous $n$ elements is called the moving average, where $n$ is called the _window size_. In this example, a kernel is implemented to compute the moving average in parallel, using the shared memory as a cache.

### Application flow

1. Define constants to control the problem size and the kernel launch parameters.
2. Allocate and initialize the input array. This array is initialized as the sequentially increasing sequence $0, 1, 2, \ldots\mod n$.
3. Allocate the device array and copy the host array to it.
4. Launch the kernel to compute the moving average.
5. Copy the result back to the host and validate it. As each average is computed using $n$ consecutive values from the input array, the average is computed over the values $0, 1, 2,\ldots, n - 1 $, the average of which is equal to $(n-1)/2$.

## Key APIs and Concepts

Device memory is allocated with `hipMalloc`, deallocated with `hipFree`. Copies to and from the device are made with `hipMemcpy` with options `hipMemcpyHostToDevice` and `hipMemcpyDeviceToHost`, respectively. A kernel is launched with the `myKernel<<<params>>>()`-syntax. Shared memory is allocated in the kernel with the `__shared__` memory space specifier.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `__shared__`
- `__syncthreads`
- `blockDim`
- `blockIdx`
- `threadIdx`

#### Host symbols

- `__global__`
- `hipFree`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipStreamDefault`
