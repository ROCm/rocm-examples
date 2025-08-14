# HIP-Doc Host Code Feature Identification Example

## Description

This example demonstrates how to use the HIP runtime API to identify device features in host code. For more
information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html#host-code-feature-identification).

### Application flow

1. The number of devices in the system is queried.
2. The first device's properties are queried.
3. Support for shared integer atomics is determined and the result printed.

## Key APIs and Concepts

* `hipGetDeviceProperties` obtains the given device's properties.
* `hipDeviceProp_t` is a struct type which is returned by `hipGetDeviceProperties`. It contains the given device's
  properties.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipGetDeviceProperties`
* `hipGetErrorString`
