# HIP-Doc Dynamic Shared Memory Example

## Description

This example demonstrates how to dynamically allocate shared memory on the host. For more information on this topic,
please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/device_memory.html#shared-memory).

### Application flow

1. The required memory size is calculated on the host.
2. The kernel is launched with an additional launch parameter: the bytes of shared memory that are required for the
   kernel.
3. The host checks for any errors.

## Key APIs and Concepts

* `extern __shared__` is used to declare a dynamically allocated shared memory symbol on the device side.
* The third launch parameter of the kernel launch contains the size (in bytes) of the shared memory to allocate.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `threadIdx`

#### Host symbols

* `hipDeviceSynchronize`
* `hipPeekAtLastError`
