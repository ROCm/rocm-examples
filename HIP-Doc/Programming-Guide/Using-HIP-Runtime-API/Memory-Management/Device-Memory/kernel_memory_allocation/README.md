# HIP-Doc Kernel Memory Allocation Example

## Description

This program demonstrates how to allocate global device memory inside a kernel. For more information on this topic,
please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/device_memory.html#global-memory).

### Application flow

1. The kernel is launched.
2. Inside the kernel, device memory is allocated using the `new` operator.
3. After computing some dummy work, the device memory is freed using the `delete` operator.
4. The host checks for any errors and then exits.

## Key APIs and Concepts

`new` and `delete` can be used inside a kernel to allocate / free global device memory.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

* `blockDim`
* `new`
* `delete`
* `__syncthreads`
* `threadIdx`

#### Host symbols

* `hipGetLastError`
