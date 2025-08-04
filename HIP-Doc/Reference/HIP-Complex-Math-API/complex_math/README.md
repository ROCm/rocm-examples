# HIP-Doc Complex Math Example

## Description

This example demonstrates how to use the HIP complex math API. For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/complex_math_api.html#hip-complex-math-api).

### Application flow

1. On the host side:
   1. Create an input vector consisting of single-precision floating-point numbers.
   2. Initialize the input vector with a simple signal: the sum of two sine waves.
   3. Create a solution vector consisting of HIP's built-in complex data type for single-precision floating-point numbers.
   4. Compute the reference solution and store it in the solution vector.
2. Then, on the device side:
   1. Create an input vector consisting of single-precision floating-point numbers.
   2. Create an output vector consisting of HIP's built-in complex data type for single-precision floating-point numbers.
   3. Copy the host's input vector to the device's input vector.
   4. Launch the kernel.
   5. Compute the DFT of the data in the device's input vector and store the result in the device's output vector.
3. Then, on the host side:
   1. Create a second output vector consisting of HIP's built-in complex data type for single-precision floating-point numbers.
   2. Copy the device's output vector to the new host-side output vector.
   3. Compare the GPU's solution to the reference solution.
4. Free the device memory.

## Key APIs and Concepts

* Use `hipMalloc` to allocate memory in the global memory of the device (GPU). This is typically necessary because
  kernels running on the device cannot access host (CPU) memory, except for device-accessible pinned host memory
  (see `hipHostMalloc`). Note that the memory returned by `hipMalloc` is uninitialized.
* Use `hipFree` to deallocate device memory previously allocated with `hipMalloc`. It is important to free memory that
  is no longer in use to prevent resource leakage.
* Use `hipMemcpy` to transfer bytes between host and device memory in both directions. A call to `hipMemcpy`
  synchronizes the device with the host, ensuring that all kernels queued before the call finish executing before the
  transfer begins. The function completes once the copying operation is finished.
* Use `hipGetErrorString` to convert a HIP error code into a human-readable string.
* Use `make_hipFloatComplex` to create HIP's complex data type for single-precision floating-point numbers.
* Use `hipCaddf` to perform the addition of two single-precision complex numbers.
* Use `hipCmulf` to perform the multiplication of two single-precision complex numbers.

## Demonstrated API calls

### HIP runtime

#### Device symbols

* `blockIdx`
* `blockDim`
* `cosf`
* `hipCaddf`
* `hipCmulf`
* `make_hipFloatComplex`
* `sinf`
* `threadIdx`

#### Host symbols

* `hipCaddf`
* `hipCmulf`
* `hipGetErrorString`
* `hipFree`
* `hipMalloc`
* `hipMemcpy`
* `make_hipFloatComplex`
