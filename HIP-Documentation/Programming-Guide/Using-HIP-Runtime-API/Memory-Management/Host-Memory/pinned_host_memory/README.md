# HIP-Documentation Pinned Host Memory Example

## Description

Pinned memory or page-locked memory is stored in pages that are locked in specific sectors in RAM and cannot be
migrated. The pointer can be used on both host and device. Accessing host-resident pinned memory in device kernels is
generally not recommended for performance, as it can force the data to traverse the host-device interconnect such as
PCIe, which is much slower than the on-device bandwidth.

The advantage of pinned memory is the improved transfer time between host and device. The disadvantage of pinned memory
is the reduced availability of RAM for other processes, which can negatively impact the overall performance of the
host.

This example demonstrates how to allocate pinned memory on the host and transfer its contents to the device.

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

* `hipHostMalloc` is used to allocate pinned memory on the host.
* `hipMalloc` is used to allocate memory on the device.
* `hipMemcpy` is used to copy memory from the host to the device and vice versa.
* `delete` is used to free pinned memory on the host.
* `hipFree` is used to free memory on the device.
* `hipGetErrorString` is used to translate a HIP error code to a human-readable error description.

## Demonstrated API Calls

### HIP Runtime

* `hipFree`
* `hipGetErrorString`
* `hipHostMalloc`
* `hipMalloc`
* `hipMemcpy`
