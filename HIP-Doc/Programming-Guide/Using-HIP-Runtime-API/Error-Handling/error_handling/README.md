# HIP-Doc Error Handling Example

## Description

HIP provides functionality to detect, report, and manage errors that occur during the execution of HIP runtime
functions or when launching kernels. Every HIP runtime function, except for launching kernels, has `hipError_t` as its
return type. `hipGetLastError()` and `hipPeekAtLastError()` can be used to catch errors from kernel launches, as these
launches do not return an error directly. HIP maintains an internal state that includes the last error code.
`hipGetLastError()` returns and resets that error to `hipSuccess`, while `hipPeekAtLastError()` simply returns the
error without changing it. To obtain a human-readable version of the errors, you can use `hipGetErrorString()` and
`hipGetErrorName()`.

This example demonstrates how to handle HIP runtime errors without creating too much code overhead.

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
