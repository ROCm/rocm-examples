# HIP-Basic Warp Shuffle Example

## Description

Kernel code for a particular block is executed in groups of threads known as a _wavefronts_ (AMD) or _warps_ (NVIDIA). Each block is is divided into as many warps as the block's size allows. If the block size is less than the warp size, then part of the warp just stays idle (as happens in this example). AMD GPUs use 64 threads per wavefront for architectures prior to RDNA™ 1. RDNA architectures support both 32 and 64 wavefront sizes.

Warps are executed in _lockstep_, i.e. all the threads in each warp execute the same instruction at the same time but with different data. This type of parallel processing is also known as Single Instruction, Multiple Data (SIMD). A block contains several warps and the warp size is dependent on the architecture, but the block size is not. Blocks and warps also differ in the way they are executed, and thus they may provide different results when used in the same piece of code. For instance, the kernel code of this example would not work as it is with block execution and shared memory access e.g. because some synchronization would be needed to ensure that every thread has written its correspondent value before trying to access it.

Higher performance in the execution of kernels can be achieved with explicit warp-level programming. This can be done by using some collective operations, known as _warp shuffles_, that allow exchanging data between threads in the same warp without the need for shared memory. This exchange occurs simultaneously for all the active threads in the warp.

This example showcases how to use the above-mentioned operations by implementing a simple matrix transpose kernel.

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

Warp shuffle is a warp-level primitive that allows for the communication between the threads of a warp. Below is a simple example that shows how the value of the thread with index 2 is copied to all other threads within the warp.
![An illustration of a single value being copied to other threads within the warp.](warp_shuffle_simple.svg)

`__shfl(var, src_lane, width = warp_size)` copies the value of a `var` from the thread `src_lane` within the warp. This operation admits a third parameter (not used in this example), `width`, defaulted to the warp size value and which allows restricting the number of threads of the warp from which values are read. Values are copied from threads with an ID in the range $[0, width-1]$. If the ID of the thread specified in the call to `__shfl` is out of that range, then the thread accessed is the one with that ID modulo `width`. The `src_lane` may also vary per thread, as shown below.

![A more complex illustration of warp shuffle, which includes a variable source.](warp_shuffle.svg)

- `hipGetDeviceProperties` gets the properties of the specified device. In this example, it is used to get the warp size of the device (GPU) used.
- `hipMalloc` allocates memory in the global memory of the device, and with `hipMemcpy` data bytes can be transferred from host to device (using `hipMemcpyHostToDevice`) or from device to host (using `hipMemcpyDeviceToHost`), among others.
- ``myKernelName<<<...>>>`` queues the execution of a kernel on a device (GPU).
- `hipGetLastError` gets the last error returned by any HIP runtime API call.
- `hipFree` deallocates device memory allocated with `hipMalloc`.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `__global__`
- `threadIdx`
- `__shfl`

#### Host symbols

- `hipFree`
- `hipGetDeviceProperties`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipStreamDefault`
