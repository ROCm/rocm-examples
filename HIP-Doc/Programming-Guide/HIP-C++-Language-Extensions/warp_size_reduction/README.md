# HIP-Doc Warp Size Reduction Example

## Description

This example demonstrates how to use HIP's built-in `warpSize` device constant for warp-level operations. For more
information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#warpsize).

This example is intended for comparison with the
[template warp size reduction example](../template_warp_size_reduction).

### Application flow

1. The device's warp size is queried.
2. The mask variables for every warp are generated.
3. An input, an output and a validation vector are created on the host.
4. A data, a mask and a result buffer are created on the device.
5. The input vector is copied to the device's data buffer.
6. A mask array is generated and copied to the device's mask buffer. Depending on the warp size queried in step 1, the
   mask will have a different type and contain different values.
7. A reduction kernel is launched.
8. The kernel launch is checked for errors.
9. The device and the host are synchronized.
10. The device's result buffer is copied to the host's output vector.
11. The output vector and the validation vector are compared.
12. The device memory is freed.

## Key APIs and Concepts

* `hipDeviceGetAttribute` is used for querying the device's warp size on the host side. Depending on its return value
  different code paths can be taken for kernel optimization purposes.
* `warpSize` is a built-in constant that can be used in device code to query the device's warp size.

## Demonstrated API calls

### HIP runtime

#### Device symbols

* `__shfl_down`
* `__syncthreads`
* `blockDim`
* `blockIdx`
* `threadIdx`
* `warpSize`

#### Host symbols

* `__builtin_popcount`
* `__builtin_popcountll`
* `hipDeviceGetAttribute`
* `hipDeviceSynchronize`
* `hipFree`
* `hipGetLastError`
* `hipMalloc`
* `hipMemcpy`
