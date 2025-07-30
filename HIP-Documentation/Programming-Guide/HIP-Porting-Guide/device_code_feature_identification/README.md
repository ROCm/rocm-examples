# HIP-Documentation Device Code Feature Identification Example

## Description

This example demonstrates how to use preprocessor macros to identify device features in device code. For more
information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html#device-code-feature-identification).

### Application flow

1. A kernel is launched called which calls a printer function.
2. The device and the host are synchronized.

## Key APIs and Concepts

* `__HIP_ARCH_HAS_DOUBLES__` is set to `1` when the target device supports double-precision floating-point numbers. It
  is set to `0` otherwise.

## Demonstrated API calls

### HIP runtime

#### Device symbols

* `__HIP_ARCH_HAS_DOUBLES__`
* `printf`

#### Host symbols

* `hipDeviceSynchronize`
