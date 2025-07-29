# HIP-Documentation Pageable Host Memory Example

## Description

Pageable memory exists in memory blocks known as "pages" that can be migrated to other storage. For example, memory can
be migrated between CPU sockets on a motherboard or in a system where the RAM runs out of space, causing it to dump
pages into the swap partition of the hard drive. This example demonstrates how to allocate pageable memory on the host
and transfer its contents to the device.

### Application flow

1. One input and one output array are created on the host, consisting of integers.
2. Both arrays are initialized, all elements in the output array are set to '0'.
3. One input and one output array are created on the device, consisting of integers.
4. The host's input array is copied to the device's input array.
5. All elements in the device's output array are set to '0'.
6. A placeholder comment simulates a kernel launch on the device.
7. The device's output array is copied to the host's output array.
8. The host memory is freed.
9. The device memory is freed.

## Key APIs and Concepts

* `new` is used to allocate pageable memory on the host.
* `hipMalloc` is used to allocate memory on the device.
* `hipMemcpy` is used to copy memory from the host to the device and vice versa.
* `delete` is used to free pageable memory on the host.
* `hipFree` is used to free memory on the device.
* `hipGetErrorString` is used to translate a HIP error code to a human-readable error description.

## Demonstrated API Calls

### HIP Runtime

* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipMemcpy`
