# HIP-Doc Pinned Host Memory Example

## Description

This example demonstrates how to allocate pinned memory on the host and transfer its contents to the device. For more
information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/host_memory.html#pinned-memory).

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
