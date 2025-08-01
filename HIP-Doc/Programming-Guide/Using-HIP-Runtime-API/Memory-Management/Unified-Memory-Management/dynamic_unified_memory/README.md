# HIP-Doc Dynamic Unified Memory Example

## Description

Unified Memory is a single memory address space accessible from any processor within a system. This single address space
is a virtual address space that abstracts the physical memory locations, enabling both the CPU and GPU to access the
same memory addresses without needing explicit data transfers. This setup simplifies memory management and enables
applications to allocate data that can be read or written on both CPUs and GPUs without explicitly copying it to a
specific CPU or GPU. This example demonstrates how to dynamically allocate unified memory and use it from both the host
and the device.

### Application flow

1. Three variables are allocated in the unified memory space, with two initialized on the host.
2. A kernel is launched on the device.
3. The kernel computes the sum of the two initialized variables and stores the result in the third variable.
4. The device and host are explicitly synchronized.
5. The result is printed.
6. The variables are freed.

## Key APIs and Concepts

* `hipDeviceSynchronize` is used to synchronize the device with the host.
* `hipFree` is used to free previously allocated unified memory.
* `hipGetErrorString` transforms a HIP error code into a human-readable string.
* `hipLaunchKernelGGL` is used to launch a kernel on the device.
* `hipMallocManaged` is used to allocate unified memory.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipLaunchKernelGGL`
* `hipMallocManaged`
