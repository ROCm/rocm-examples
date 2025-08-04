# HIP-Doc Ordinary Memory Allocation Example

## Description

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
