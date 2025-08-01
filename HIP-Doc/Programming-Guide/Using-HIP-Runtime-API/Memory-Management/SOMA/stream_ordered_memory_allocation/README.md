# HIP-Doc Stream Ordered Memory Allocation Example

## Description

The Stream Ordered Memory Allocator (SOMA) is part of the HIP runtime API. SOMA provides an asynchronous memory
allocation mechanism with stream-ordering semantics. You can use SOMA to allocate and free memory in stream order,
which ensures that all asynchronous accesses occur between the stream executions of allocation and deallocation.
Compliance with stream order prevents use-before-allocation or use-after-free errors, which helps to avoid an undefined
behavior.

This example demonstrates how to use stream ordered memory allocations.

### Prerequisites

The Stream Ordered Memory Allocator API is currently under development for Windows. For the time being, this example
only works on Linux.

### Application flow

1. HIP is initialized.
2. A data array is allocated using the SOMA API.
3. A compute kernel is launched on the device.
4. The result is copied to the host.
5. The result is printed.
6. The array is freed using the SOMA API.
7. Host and device are synchronized.

## Key APIs and Concepts

* `hipDeviceSynchronize` is used to synchronize the device with the host.
* `hipMallocAsync` is a SOMA API call used to allocate memory.
* `hipMemcpy` transfers bytes between the host and the device.
* `hipFreeAsync` is a SOMA API call used to free previously allocated memory.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipFreeAsync`
* `hipMallocAsync`
* `hipMemcpy`
