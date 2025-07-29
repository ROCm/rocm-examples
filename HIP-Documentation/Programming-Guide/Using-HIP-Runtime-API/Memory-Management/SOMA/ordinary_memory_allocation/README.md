# HIP-Documentation Ordinary Memory Allocation Example

## Description

The Stream Ordered Memory Allocator (SOMA) is part of the HIP runtime API. SOMA provides an asynchronous memory
allocation mechanism with stream-ordering semantics. You can use SOMA to allocate and free memory in stream order,
which ensures that all asynchronous accesses occur between the stream executions of allocation and deallocation.
Compliance with stream order prevents use-before-allocation or use-after-free errors, which helps to avoid an undefined
behavior.

This example demonstrates an ordinary memory allocation and should be compared to the
[SOMA example](../stream_ordered_memory_allocation/).

### Application flow

1. HIP is initialized.
2. A data array is allocated.
3. A compute kernel is launched on the device.
4. The result is copied to the host.
5. The result is printed.
6. The array is freed.
7. Host and device are synchronized.

## Key APIs and Concepts

* `hipDeviceSynchronize` is used to synchronize the device with the host.
* `hipMalloc` is a HIP API call used to allocate memory.
* `hipMemcpy` transfers bytes between the host and the device.
* `hipFree` is a HIP API call used to free previously allocated memory.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipMalloc`
* `hipMemcpy`
