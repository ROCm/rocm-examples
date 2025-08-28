# HIP-Doc Unified Memory Advice Example

## Description

This example demonstrates how to set unified memory runtime hints. For more information on this topic, please refer to
the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html#memory-advice).

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
* `hipMallocManaged` is used to allocate unified memory.
* `hipMemAdvise` is used to set advice for variables in the unified memory space.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipMallocManaged`
* `hipMemAdvise`
