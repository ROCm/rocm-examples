# HIP-Doc Data Prefetching Example

## Description

This example demonstrates how to prefetch data in the unified memory space before it is actually needed. For more
information, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html#data-prefetching).

### Application flow

1. The current device ID is obtained by the host.
2. Memory for three variables is allocated in the unified memory space.
3. Two of the variables are initialized by the host.
4. Prefetch instructions are issued for all three variables.
5. A kernel is launched on the device.
6. The kernel computes the sum of the two initialized variables and stores the result in the third variable.
7. A prefetch instruction is issued for the result variable.
8. Host and device are explicitly synchronized.
9. The result is printed.
10. The variables are freed.

## Key APIs and Concepts

* `hipDeviceSynchronize` is used to synchronize the device with the host.
* `hipFree` is used to free previously allocated unified memory.
* `hipGetErrorString` transforms a HIP error code into a human-readable string.
* `hipMallocManaged` is used to allocate unified memory.
* `hipMemPrefetchAsyc` is used to move data to the device before it is needed.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipMallocManaged`
* `hipMemPrefetchAsync`
