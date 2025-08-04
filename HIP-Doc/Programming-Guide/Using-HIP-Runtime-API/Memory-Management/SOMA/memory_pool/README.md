# HIP-Doc Memory Pool Example

## Description

This example demonstrates how to use the stream ordered memory allocation (SOMA) API to set up and manage a memory
pool. For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html#memory-pools).

### Prerequisites

The Stream Ordered Memory Allocator API is currently under development for Windows. For the time being, this example
only works on Linux.

### Application flow

1. A HIP stream is created.
2. A HIP memory pool is created.
3. A data array is allocated using memory from the pool.
4. A compute kernel is launched on the device.
5. The stream is synchronized.
6. The result is copied to the host.
7. The result is printed.
8. The array is freed and the memory returned to the pool.
9. The stream is synchronized.
10. The memory pool and the stream are destroyed.

## Key APIs and Concepts

* `hipStreamCreate` creates a HIP stream.
* `hipMemPoolCreate` creates a memory pool.
* `hipMallocFromPoolAsync` uses the SOMA API to allocate memory from the pool.
* `hipStreamSynchronize` is used to synchronize the stream with the host.
* `hipMemcpy` transfers bytes between the host and the device.
* `hipFreeAsync` is a SOMA API call used to free memory and (in this case) return it to the memory pool.
* `hipMemPoolDestroy` destroys a memory pool.
* `hipStreamDestroy` destroys a HIP stream.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipFreeAsync`
* `hipMallocFromPoolAsync`
* `hipMemPoolCreate`
* `hipMemPoolDestroy`
* `hipMemcpy`
* `hipStreamCreate`
* `hipStreamDestroy`
* `hipStreamSynchronize`
