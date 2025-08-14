# HIP-Doc Explicit Memory Example

## Description

This example demonstrates how to perform explicit memory management by allocating memory on the device and transferring
bytes between the host and the device. It should be compared to the various unified memory examples in this example's
[parent folder](..).

### Application flow

1. Three variables are defined on the host, with two initialized.
2. Three variables are allocated on the device.
3. The initialized variables' data are copied from the host to the corresponding device variables.
4. A kernel is launched on the device.
5. The kernel computes the sum of the two initialized variables and stores the result in the third variable.
6. The result is copied from the device to the third host variable.
7. The device variables are freed.
8. The result is printed.

## Key APIs and Concepts

* `hipFree` is used to free previously allocated device memory.
* `hipGetErrorString` transforms a HIP error code into a human-readable string.
* `hipLaunchKernelGGL` is used to launch a kernel on the device.
* `hipMalloc` is used to allocate device memory.
* `hipMemcpy` is used to transfer bytes between the host and the device.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipLaunchKernelGGL`
* `hipMalloc`
* `hipMemcpy`
