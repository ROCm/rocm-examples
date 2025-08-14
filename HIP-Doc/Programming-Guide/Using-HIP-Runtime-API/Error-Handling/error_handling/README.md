# HIP-Doc Error Handling Example

## Description

This example demonstrates how to handle HIP runtime errors without creating too much code overhead. For more information
on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/error_handling.html).

### Application flow

1. Two input vectors and one output vector are created by the host. The input vectors are initialized.
2. Two input buffers and one output buffer are created on the device. The HIP API calls are checked for errors.
3. The host's input vectors are copied to the device's input buffers. The HIP API calls are checked for errors.
4. A computation kernel is launched on the device.
5. An explicit check for errors during the kernel launch and execution is performed.
6. The host and the device are synchronized. The HIP API call is checked for errors.
7. The result is copied back to the host's output vector. The HIP API call is checked for errors.
8. The device buffers are freed. The HIP API calls are checked for errors.
9. The result is printed.

## Key APIs and Concepts

* `hipGetErrorString` transforms a HIP error code into a human-readable string.
* `hipGetLastError` returns the last error encountered by the HIP runtime. This can be used to check for errors during
  kernel execution.

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
* `hipGetLastError`
* `hipLaunchKernelGGL`
* `hipMalloc`
* `hipMemcpy`
