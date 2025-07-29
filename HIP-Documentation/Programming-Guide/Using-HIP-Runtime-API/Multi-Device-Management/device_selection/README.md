# HIP-Documentation Device Selection Example

## Description

Once you have [enumerated the available GPUs](../device_enumeration/), the next step is to select a specific device for
computation. This involves setting the active GPU that will execute subsequent operations. This step is crucial in
multi-GPU systems where different GPUs might have different capabilities or workloads. By selecting the appropriate
device, you ensure that the computational tasks are directed to the correct GPU, optimizing performance and resource
utilization.

This example demonstrates how to switch between the different devices in the system and assign work to them.

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
