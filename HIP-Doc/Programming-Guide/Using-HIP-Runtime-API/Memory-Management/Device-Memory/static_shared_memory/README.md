# HIP-Doc Static Shared Memory Example

## Description

This example demonstrates how to statically allocate shared memory inside a kernel. For more information on this topic,
please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/device_memory.html#shared-memory).

### Application flow

1. The kernel is launched.
2. The host checks for any errors.

## Key APIs and Concepts

* `__shared__` is used to declare a statically allocated shared memory symbol on the device side.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipPeekAtLastError`
