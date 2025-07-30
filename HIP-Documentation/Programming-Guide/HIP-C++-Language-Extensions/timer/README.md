# HIP-Documentation Timer Example

## Description

This example demonstrates how to use HIP's built-in timer from within the kernel and how to query the wall clock rate
on the host. For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#timer-functions).

### Application flow

1. The device's wall clock rate is queried.
2. A kernel is launched which queries the device's cycle count twice and calculates the difference between both values.
3. The host and the device are synchronized.
4. The wall clock rate is printed.

## Key APIs and Concepts

* `hipDeviceGetAttribute` is used for querying the device's wall clock rate on the host side.
* `clock64` is a built-in device function that returns the device's cycle count at the time of the call.

## Demonstrated API calls

### HIP runtime

#### Device symbols

* `clock64`

#### Host symbols

* `hipDeviceGetAttribute`
* `hipDeviceSynchronize`
* `hipGetErrorString`
