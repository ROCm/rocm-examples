# HIP-Documentation Identifying Host and Device Compilation Pass Example

## Description

This example demonstrates how to use preprocessor macros to distinguish between the host and the device compilation
pass(es). For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html#identifying-host-or-device-compilation-pass).

### Application flow

1. A kernel is launched called which calls a printer function.
2. The device and the host are synchronized.
3. The printer function is called on the host.

## Key APIs and Concepts

* `__HIP_DEVICE_COMPILE__` is defined whenever a device compilation pass takes place. It is undefined otherwise.

## Demonstrated API calls

### HIP runtime

#### Device symbols

* `__HIP_DEVICE_COMPILE__`
* `printf`

#### Host symbols

* `hipDeviceSynchronize`
