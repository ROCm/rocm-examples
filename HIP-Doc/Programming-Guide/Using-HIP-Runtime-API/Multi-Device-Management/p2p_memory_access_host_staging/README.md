# HIP-Doc Peer-to-Peer Memory Access with Host Staging Example

## Description

This example demonstrates how to copy data between devices by adding peer-to-peer accesses to the
[device selection example](../device_selection/), but explicitly does not enable peer-to-peer access for the devices.
For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html#peer-to-peer-memory-access).

This version is intended for comparison with the [default peer-to-peer example](../p2p_memory_access).

### Application flow

1. The number of HIP devices is queried and an error printed if there are less than two in the system.
2. Tasks are performed sequentially on both devices:
    1. The device is set as active device.
    2. Memory is allocated.
    3. A kernel is launched on the device.
    4. The host and the device are synchronized.
3. The first device is set as active device.
4. The second device's result is copied to the first device's result buffer.
5. The first device's result buffer is copied to the host.
6. The second device is set as active device.
7. The second device's result buffer is copied to the host.
8. Both results are printed.
9. Memory on both devices is freed.

## Key APIs and Concepts

* `hipDeviceEnablePeerAccess` enables peer-to-peer memory access for the given device.
  **It is deliberately omitted in this example.**

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `threadIdx`

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipMemcpy`
* `hipSetDevice`
