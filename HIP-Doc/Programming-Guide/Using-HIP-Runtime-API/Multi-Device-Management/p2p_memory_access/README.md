# HIP-Doc Peer-to-Peer Memory Access Example

## Description

This example demonstrates how to copy data between devices by adding peer-to-peer accesses to the
[device selection example](../device_selection/). For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html#peer-to-peer-memory-access).

This example is intended for comparison with the
[failed peer-to-peer memory access example](../p2p_memory_access_failed).

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
