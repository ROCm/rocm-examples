# HIP-Doc Dynamic Unified Memory Example

## Description

This example demonstrates how to dynamically allocate unified memory and use it from both the host and the device. For
more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html).

### Application flow

1. Three variables are allocated in the unified memory space, with two initialized on the host.
2. A kernel is launched on the device.
3. The kernel computes the sum of the two initialized variables and stores the result in the third variable.
4. The device and host are explicitly synchronized.
5. The result is printed.
6. The variables are freed.

## Key APIs and Concepts

* `hipDeviceGetAttribute` is used to query the device's ability to allocate managed memory.
* `hipDeviceSynchronize` is used to synchronize the device with the host.
* `hipFree` is used to free previously allocated unified memory.
* `hipGetErrorString` transforms a HIP error code into a human-readable string.
* `hipMallocManaged` is used to allocate unified memory.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceGetAttribute`
* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipMallocManaged`
