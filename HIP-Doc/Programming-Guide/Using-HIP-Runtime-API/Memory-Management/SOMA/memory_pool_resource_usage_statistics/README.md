# HIP-Doc Memory Pool Resource Usage Statistics Example

## Description

This example demonstrates how to query resource usage statistics for a memory pool. For more information on this topic,
please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html#resource-usage-statistics).

### Prerequisites

The Stream Ordered Memory Allocator API is currently under development for Windows. For the time being, this example
only works on Linux.

### Application flow

1. The HIP device is initialized.
2. A handle for the default memory pool is acquired.
3. Memory is allocated from the pool.
4. The allocated memory is freed.
5. The pool is trimmed to a new size.
6. The pool resource usage statistics are queried and printed.
7. The statistics are reset.
8. The pool resource usage statistics are queried and printed again.
9. The pool handle is destroyed.

## Key APIs and Concepts

* `hipSetDevice` sets the active HIP device.
* `hipDeviceGetDefaultMemPool` obtains the handle for the device's default memory pool.
* `hipMalloc` allocates memory from the pool.
* `hipFree` frees memory and (in this case) returns it to the memory pool.
* `hipMemPoolTrimTo` trims the pool to a new size.
* `hipMemPoolGetAttribute` is used to query the pool for individual resource attributes.
* `hipMemPoolSetAttribute` overwrites a pool's (resource) attribute.
* `hipMemPoolDestroy` destroys a memory pool.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceGetDefaultMemPool`
* `hipFree`
* `hipMalloc`
* `hipMemPoolDestroy`
* `hipMemPoolGetAttribute`
* `hipMemPoolSetAttribute`
* `hipMemPoolTrimTo`
* `hipSetDevice`
