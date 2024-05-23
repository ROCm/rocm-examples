# HIP-Basic Hello World Example

## Description

This example showcases launching kernels and printing from device programs.

### Application flow

1. A kernel is launched: function `hello_world_kernel` is executed on the device. This function uses the coordinate built-ins to print a unique identifier from each thread.
2. _Synchronization_ is performed: the host program execution halts until all kernels on the device have finished executing.

## Key APIs and Concepts

- `myKernelName<<<gridDim, blockDim, dynamicShared, stream>>>(kernelArguments)` launches a kernel. In other words: it calls a function marked with `__global__` to execute on the device. An _execution configuration_ is specified, which are the grid and block dimensions, the amount of additional shared memory to allocate, and the stream where the kernel should execute. Optionally, the kernel function may take arguments as well.

- `hipDeviceSynchronize` synchronizes with the device, halting the host until all commands associated with the device have finished executing.

- Printing from device functions is performed using `printf`.

- Function-type qualifiers are used to indicate the type of a function.

  - `__global__` functions are executed on the device and called from the host.
  - `__device__` functions are executed on the device and called from the device only.
  - `__host__` functions are executed on the host and called from the host.
  - Functions marked with both `__device__` and `__host__` are compiled for host and device. This means that these functions cannot contain any device- or host-specific code.

- Coordinate built-ins determine the coordinate of the active work item in the execution grid.

  - `threadIdx` is the 3D coordinate of the active work item in the block of threads.
  - `blockIdx` is the 3D coordinate of the active work item in the grid of blocks.

## Demonstrated API Calls

### HIP Runtime

- `hipDeviceSynchronize`
- `__device__`
- `__global__`
- `__host__`
- `threadIdx`
- `blockIdx`

## Supported Platforms

Windows is currently not supported by the hello world example, due to a driver failure with `printf` from device code.
