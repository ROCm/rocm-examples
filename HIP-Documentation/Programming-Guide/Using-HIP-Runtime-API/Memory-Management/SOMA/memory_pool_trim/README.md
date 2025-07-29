# HIP-Documentation Memory Pool Trim Example

## Description

Memory pools provide a way to manage memory with stream-ordered behavior while ensuring proper synchronization and
avoiding memory access errors. Division of a single memory system into separate pools facilitates querying the access
path properties for each partition. Memory pools are used for host memory, device memory, and unified memory.

To improve performance, it is a good practice to adjust the memory pool size using `hipMemPoolTrimTo()`. It helps to
reclaim memory from an excessive memory pool, which optimizes memory usage for your application.

This example demonstrates how to trim a memory pool to a new size.

### Prerequisites

The Stream Ordered Memory Allocator API is currently under development for Windows. For the time being, this example
only works on Linux.

### Application flow

1. The HIP device is initialized.
2. A handle for the default memory pool is acquired.
3. Memory is allocated from the pool.
4. The allocated memory is freed.
5. The pool is trimmed to a new size.
6. The pool handle is destroyed.

## Key APIs and Concepts

* `hipSetDevice` sets the active HIP device.
* `hipDeviceGetDefaultMemPool` obtains the handle for the device's default memory pool.
* `hipMalloc` allocates memory from the pool.
* `hipFree` frees memory and (in this case) returns it to the memory pool.
* `hipMemPoolTrimTo` trims the pool to a new size.
* `hipMemPoolDestroy` destroys a memory pool.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceGetDefaultMemPool`
* `hipFree`
* `hipMalloc`
* `hipMemPoolDestroy`
* `hipMemPoolTrimTo`
* `hipSetDevice`
