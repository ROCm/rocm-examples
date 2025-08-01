# HIP-Doc 8-bit Floating-Point Example

## Description

Modern computing tasks often require a balance between numerical precision, hardware resources, and processing speed.
Low-precision floating-point formats in HIP, such as FP8 (Quarter Precision) and FP16 (Half Precision), reduce memory
and bandwidth requirements compared to traditional 32-bit and 64-bit formats. This example illustrates a subset of HIP's
functionality for 8-bit floating-point numbers. The programming approach reflects the common theme of converting
low-precision types to 32-bit floating-point numbers and back.

### Application flow

1. Query the GPU device properties and check for a supported GPU architecture.
2. On the host side:
   1. Create an input vector and an output vector consisting of 32-bit floating-point numbers.
   2. Initialize the input vector.
   3. Convert each element in the input vector to an 8-bit floating-point number.
   4. Convert the resulting 8-bit floating-point numbers to 32-bit floating-point numbers and store them in the output
      vector.
3. On the device side:
   1. Create an input vector and an output vector consisting of 32-bit floating-point numbers.
   2. Copy the host's input vector to the device's input vector.
   3. Launch the kernel.
   4. Convert each element in the device's input vector to an 8-bit floating-point number within the kernel.
   5. Convert the resulting 8-bit floating-point numbers to 32-bit floating-point numbers and store them in the device's
      output vector.
4. On the host side:
   1. Create a second output vector consisting of 32-bit floating-point numbers.
   2. Copy the device's output vector to the new host-side output vector.
5. Free the device memory.
6. Validate the second output vector, which contains the results obtained on the GPU, against the first output vector.

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
* Use `hipGetDeviceProperties` to query the GPU's information and check for a supported GPU architecture.
* Use `__hip_cvt_float_to_fp8` to convert a 32-bit floating-point number to its 8-bit equivalent.

## Demonstrated API calls

### HIP runtime

#### Device symbols

* `threadIdx`
* `__hip_cvt_float_to_fp8`

#### Host symbols

* `hipGetDeviceProperties`
* `hipGetErrorString`
* `hipMalloc`
* `hipMemcpy`
* `hipFree`
* `__hip_cvt_float_to_fp8`
