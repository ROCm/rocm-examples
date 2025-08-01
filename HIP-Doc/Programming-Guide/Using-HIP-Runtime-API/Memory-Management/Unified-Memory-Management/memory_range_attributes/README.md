# HIP-Doc Memory Range Attributes Example

## Description

This example demonstrates how to query attributes of a given memory range.

### Prerequisites

The `hipMemRangeGetAttribute` API is currently under development for Windows. For the time being, this example only
works on Linux.

### Application flow

1. The current device ID is obtained by the host.
2. Memory for three variables is allocated in the unified memory space.
3. The two input variables are initialized.
4. A memory advice is set for the first input variable.
5. A kernel is launched on the device.
6. The kernel computes the sum of the two initialized variables and stores the result in the third variable.
7. Host and device are explicitly synchronized.
8. An attribute of the first variable's memory range is queried.
9. The computation result is printed.
10. The queried memory range attribute is printed.
11. The variables are freed.

## Key APIs and Concepts

* `hipDeviceSynchronize` is used to synchronize the device with the host.
* `hipFree` is used to free previously allocated unified memory.
* `hipGetErrorString` transforms a HIP error code into a human-readable string.
* `hipLaunchKernelGGL` is used to launch a kernel on the device.
* `hipMallocManaged` is used to allocate unified memory.
* `hipMemAdvise` is used to set advice for variables in the unified memory space.
* `hipMemRangeGetAttribute` is used to query for memory range attributes.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipLaunchKernelGGL`
* `hipMallocManaged`
* `hipMemAdvise`
* `hipMemRangeGetAttribute`
