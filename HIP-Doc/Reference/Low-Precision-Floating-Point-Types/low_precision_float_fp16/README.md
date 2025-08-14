# HIP-Doc Half-Precision Floating-Point Example

## Description

This example demonstrates how to use half-precision floating-point numbers. For more information on this topic, please
refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html#fp16-half-precision).

### Application flow

1. Create and initialize input vectors containing 32-bit floating-point numbers on the host.
2. Compute the expected results on the host.
3. Allocate memory for input vectors (containing 16-bit floating-point numbers) and output vectors (containing 32-bit
floating-point numbers) on the device.
4. Convert the input vectors to 16-bit floating-point vectors.
5. Copy the input vectors from the host to the device.
6. Launch the device kernel.
7. For each pair of 16-bit elements in the input vectors, the kernel computes the 16-bit sum and converts it to 32-bit.
8. Store the sum as a 32-bit floating-point number in the device's output vector.
9. Copy the result back to the host.
10. Free the device-side memory.
11. Validate the device-side result against the host-side result from step 2.

## Key APIs and Concepts

## Demonstrated API calls

* Use `hipMalloc` to allocate memory in the global memory of the device (GPU). This is typically necessary because
  kernels running on the device cannot access host (CPU) memory, except for device-accessible pinned host memory (see
  `hipHostMalloc`). Note that the memory returned by `hipMalloc` is uninitialized.
* Use `hipFree` to deallocate device memory previously allocated with `hipMalloc`. It is important to free memory that
  is no longer in use to prevent resource leakage.
* Use `hipMemcpy` to transfer bytes between host and device memory in both directions. A call to `hipMemcpy`
  synchronizes the device with the host, ensuring that all kernels queued before the call finish executing before the
  transfer begins. The function completes once the copying operation is finished.
* Use `hipGetErrorString` to convert a HIP error code into a human-readable string.
* Use `__float2half` on the host side to convert 32-bit floating-point values to their 16-bit equivalents.
* Use `__half2float` on the device side to convert 16-bit floating-point values to their 32-bit equivalents.

### HIP runtime

#### Device symbols

* `threadIdx`
* `__half2float`

#### Host symbols

* `hipGetErrorString`
* `hipMalloc`
* `hipMemcpy`
* `hipFree`
* `__float2half`
