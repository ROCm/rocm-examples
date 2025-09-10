# HIP-Doc Standard Unified Memory Example

## Description

This example demonstrates how to dynamically allocate unified memory with standard C++ facilities and use it from both
the host and the device. For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html).

### Prerequisites

This example is only supported on Linux systems with Heterogeneous Memory Management (HMM) support. In addition, the
environment variable `HSA_XNACK` must be set to `1`.

### Application flow

1. Three variables are allocated in the unified memory space, with two initialized on the host.
2. A kernel is launched on the device.
3. The kernel computes the sum of the two initialized variables and stores the result in the third variable.
4. The device and host are explicitly synchronized.
5. The result is printed.
6. The variables are freed.

## Key APIs and Concepts

* `hipDeviceSynchronize` is used to synchronize the device with the host.
* `hipGetErrorString` transforms a HIP error code into a human-readable string.
* The standard C++ operators `new` and `delete` are used to manage unified memory allocations.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipGetErrorString`
