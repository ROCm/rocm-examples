# HIP-Programming-Guide Peer-to-Peer Memory Access Example

## Description

In multi-GPU systems, peer-to-peer memory access enables one GPU to directly read or write to the memory of another GPU.
Enabling peer-to-peer access can significantly improve the performance of applications that require frequent data
exchange between GPUs, as it eliminates the need to transfer data through the host memory.

This example demonstrates how to copy data between devices by adding peer-to-peer accesses to the
[device selection example](../device_selection/).

### Application flow

1. The number of HIP devices is queried and an error printed if there are less than two in the system.
2. Two devices are initialized sequentially:
    1. The device is set as active device.
    2. Peer-to-peer accesses are enabled for the device.
3. Tasks are performed sequentially on both devices:
    1. The device is set as active device.
    2. Memory is allocated.
    3. A kernel is launched on the device.
    4. The host and the device are synchronized.
4. The first device is set as active device.
5. The second device's result is copied to the first device's result buffer.
6. The first device's result buffer is copied to the host.
7. The second device is set as active device.
8. The second device's result buffer is copied to the host.
9. Both results are printed.
10. Memory on both devices is freed.

## Key APIs and Concepts

* `hipDeviceEnablePeerAccess` enables peer-to-peer memory access for the given device.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `threadIdx`

#### Host symbols

* `hipDeviceEnablePeerAccess`
* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipMemcpy`
* `hipSetDevice`
