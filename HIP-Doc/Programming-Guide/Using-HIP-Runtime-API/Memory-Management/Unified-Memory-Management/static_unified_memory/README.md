# HIP-Doc Static Unified Memory Example

## Description

This example demonstrates how to statically allocate unified memory and use it from both the host and the device. For
more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html#id1).

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
