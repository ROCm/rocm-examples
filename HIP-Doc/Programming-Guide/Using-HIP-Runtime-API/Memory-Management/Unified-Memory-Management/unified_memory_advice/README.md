# HIP-Doc Unified Memory Advice Example

## Description

Unified Memory is a single memory address space accessible from any processor within a system. This single address space
is a virtual address space that abstracts the physical memory locations, enabling both the CPU and GPU to access the
same memory addresses without needing explicit data transfers. This setup simplifies memory management and enables
applications to allocate data that can be read or written on both CPUs and GPUs without explicitly copying it to a
specific CPU or GPU. This example demonstrates how to set unified memory runtime hints; this is a technique that can be
used to improve an application's performance.

### Application flow

1. The current device ID is obtained by the host.
2. Memory for three variables is allocated in the unified memory space.
3. Memory advice is set for all three variables.
4. The two input variables are initialized.
5. A kernel is launched on the device.
6. The kernel computes the sum of the two initialized variables and stores the result in the third variable.
7. Host and device are explicitly synchronized.
8. The result is printed.
9. The variables are freed.

## Key APIs and Concepts

* `hipDeviceSynchronize` is used to synchronize the device with the host.
* `hipFree` is used to free previously allocated unified memory.
* `hipGetErrorString` transforms a HIP error code into a human-readable string.
* `hipLaunchKernelGGL` is used to launch a kernel on the device.
* `hipMallocManaged` is used to allocate unified memory.
* `hipMemAdvise` is used to set advice for variables in the unified memory space.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipLaunchKernelGGL`
* `hipMallocManaged`
* `hipMemAdvise`
