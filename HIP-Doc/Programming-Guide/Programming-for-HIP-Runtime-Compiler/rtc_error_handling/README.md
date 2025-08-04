# HIP-Doc HIPRTC Error Handling Example

## Description

This example demonstrates how to check the HIPRTC API calls for errors. For more information on this topic,
please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html#error-handling).

### Application flow

1. A HIPRTC program handle is created from a string which contains invalid HIP kernel source code.
2. The device's properties are queried in order to obtain the correct GPU architecture.
3. The HIPRTC program is compiled for the queried architecture.
4. Since the kernel code is invalid, an error code is returned.
5. The error code is transformed into a human-readable string. The string is printed.
6. The HIPRTC program handle is destroyed.

## Key APIs and Concepts

* `hiprtcResult` is a type which can take several values. Unless the value is `HIPRTC_SUCCESS` an error has occurred.
* `hiprtcGetErrorString` transforms a `hiprtcResult` value into a human-readable string.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipGetDeviceProperties`
* `hiprtcCompileProgram`
* `hiprtcCreateProgram`
* `hiprtcDestroyProgram`
* `hiprtcGetErrorString`
