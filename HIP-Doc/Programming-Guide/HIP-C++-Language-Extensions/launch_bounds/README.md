# HIP-Doc Launch Bounds Example

## Description

This example demonstrates how to call global device functions (kernels) with launch bounds. For more information on this
topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#launch-bounds).

### Application flow

1. A device buffer is allocated.
2. A kernel is launched which consumes the device buffer.
3. The host and the device are synchronized.
4. The device memory is freed.

## Key APIs and Concepts

* Use `hipMalloc` to allocate memory in the global memory of the device (GPU). This is typically necessary because
  kernels running on the device cannot access host (CPU) memory, except for device-accessible pinned host memory (see
`hipHostMalloc`). Note that the memory returned by `hipMalloc` is uninitialized.
* Use `hipFree` to deallocate device memory previously allocated with `hipMalloc`. It is important to free memory that
  is no longer in use to prevent resource leakage.
* Use the triple chevron syntax `kernel_name<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(parameters)` to launch a
  kernel on the device.
* Use `hipDeviceSynchronize` to synchronize the host and the device. This is a blocking call which only returns once all
  outstanding device operations have finished.
* Use `hipGetErrorString` to convert a HIP error code into a human-readable string.

## Demonstrated API calls

### HIP runtime

#### Device symbols

* `blockIdx`
* `blockDim`
* `threadIdx`

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
