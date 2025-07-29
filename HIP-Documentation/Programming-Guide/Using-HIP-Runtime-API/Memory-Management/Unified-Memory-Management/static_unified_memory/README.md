# HIP-Documentation Static Unified Memory Example

## Description

Unified Memory is a single memory address space accessible from any processor within a system. This setup simplifies
memory management and enables applications to allocate data that can be read or written on both CPUs and GPUs without
explicitly copying it to a specific CPU or GPU. This example demonstrates how to statically allocate unified memory
and use it from both the host and the device.

### Application flow

1. Three variables are globally declared in the unified memory space.
2. Two of these variables are initialized on the host.
3. A kernel is launched on the device.
4. The kernel computes the sum of the two initialized variables and stores the result in the third variable.
5. The device and host are explicitly synchronized.
6. The result is printed.

## Key APIs and Concepts

* `hipDeviceSynchronize` is used to synchronize the device with the host.
* `hipGetErrorString` transforms a HIP error code into a human-readable string.
* `hipLaunchKernelGGL` is used to launch a kernel on the device.
* The `__managed__` attribute is used to statically allocate memory in the unified memory space.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipGetErrorString`
* `hipLaunchKernelGGL`
