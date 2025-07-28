# HIP-Programming-Guide Math Example

## Description

HIP provides device-callable math operations, supporting most math functions available in standard C++. This example
shows a simplified method for computing ULP (units in the last place) differences between HIP and standard C++ math
functions by first finding where the maximum absolute error occurs.

### Application flow

1. Output vectors are created on the host and the device, consisting of single-precision floating-pointing numbers.
2. A kernel is launched and its output written to the device's output vector.
3. The results are copied to the host's output vector.
4. The calculation performed by the kernel is performed again by the host. The results are compared to the ones
   obtained from the device.
5. The ULP difference is calculated.
6. The results are printed.
7. The device memory is freed.

## Key APIs and Concepts

* Use `hipMalloc` to allocate memory in the global memory of the device (GPU). This is typically necessary because
  kernels running on the device cannot access host (CPU) memory, except for device-accessible pinned host memory (see
`hipHostMalloc`). Note that the memory returned by `hipMalloc` is uninitialized.
* Use `hipFree` to deallocate device memory previously allocated with `hipMalloc`. It is important to free memory that
  is no longer in use to prevent resource leakage.
* Use `hipMemcpy` to transfer bytes between host and device memory in both directions. A call to `hipMemcpy`
  synchronizes the device with the host, ensuring that all kernels queued before the call finish executing before the
  transfer begins. The function completes once the copying operation is finished.
* Use `hipGetErrorString` to convert a HIP error code into a human-readable string.
* Use `hipPeekAtLastError` to retrieve the last error returned by any HIP runtime call.

## Demonstrated API calls

### HIP runtime

#### Device symbols

* `blockIdx`
* `blockDim`
* `sinf`
* `threadIdx`

#### Host symbols

* `hipGetErrorString`
* `hipFree`
* `hipMalloc`
* `hipMemcpy`
* `hipPeekAtLastError`
