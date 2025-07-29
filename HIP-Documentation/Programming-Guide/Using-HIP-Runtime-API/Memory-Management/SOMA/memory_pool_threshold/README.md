# HIP-Documentation Memory Pool Threshold Example

## Description

Memory pools provide a way to manage memory with stream-ordered behavior while ensuring proper synchronization and
avoiding memory access errors. Division of a single memory system into separate pools facilitates querying the access
path properties for each partition. Memory pools are used for host memory, device memory, and unified memory.

This example demonstrates how to use the stream ordered memory allocation (SOMA) API to set up and manage a memory
pool, while defining a threshold to specify an amount of memory to reserve.

### Prerequisites

The Stream Ordered Memory Allocator API is currently under development for Windows. For the time being, this example
only works on Linux.

### Application flow

1. A HIP stream is created.
2. A HIP memory pool is created and a threshold defined which specifies the amount of reserved memory to hold onto.
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
* `hipMemPoolSetAttribute` sets the pool's threshold attribute.
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
* `hipMemPoolSetAttribute`
* `hipMemcpy`
* `hipStreamCreate`
* `hipStreamDestroy`
* `hipStreamSynchronize`
