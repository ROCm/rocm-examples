# HIP-Doc Device Selection Example

## Description

This example demonstrates how to switch between the different devices in the system and assign work to them. For more
information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html#device-selection).

### Application flow

1. The first device is set as active device.
2. Memory is allocated.
3. A kernel is launched.
4. The host and the first device are synchronized.
5. The second device is set as active device.
6. Memory is allocated.
7. A kernel is launched.
8. The host and the second device are synchronized.
9. The first device is set as active device.
10. The kernel's results are copied from the first device to the host.
11. The second device is set as active device.
12. The kernel's results are copied from the second device to the host.
13. The results from both devices are printed.
14. The memory on both devices is freed.

## Key APIs and Concepts

* `hipSetDevice` sets the active device. All succeeding HIP API calls will target this device until `hipSetDevice` is
  called again or the application exits.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `threadIdx`

#### Host symbols

* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipMemcpy`
* `hipSetDevice`
