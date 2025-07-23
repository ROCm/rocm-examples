# hipFFT Callback Example

## Description

This example illustrates the use of hipFFT `callback` functionality. It shows how to use load callback, a user-defined callback function that is run to load input from global memory at the start of the transform, with hipFFT. At the time of writing this example, the `callback` functionality is not yet supported neither on Windows or CUDA.

### Application flow

1. Allocate and initialize the host input data and filter.
2. Allocate device memory and copy host input data and filter from host to device.
3. Create an 1D FFT plan.
4. Allocate and initialize callback data on host.
5. Copy callback data from host to device.
6. Set the callback with the callback data and device function.
7. Execute FFT plan which multiplies each element by filter element and scales.
8. Copy the results from device to host and print it.
9. Destroy plan and free device memory.

## Key APIs and Concepts

### hipFFT

- The `hipfftHandle` needs to be created with `hipfftCreate(...)` before use and destroyed with `hipfftDestroy(...)` after use.
- The callback needs to be set with `hipfftXtSetCallback` which is set on the `hipfftHandle`, with a pointer to the function, the callback type and the callback data.
- The type `hipfftDoubleComplex` is used to have double-precision for single-precision `hipfftComplex` can be used.
- The transform type `HIPFFT_Z2Z` is used to have double-precision for single-precision `HIPFFT_C2C` can be used.
- To execute the plan `hipfftExecZ2Z` is used for double-precision for single-precision `hipfftExecC2C` can be used.

## Used API surface

### hipFFT

- `hipfftCreate`
- `hipfftDestroy`
- `hipfftDoubleComplex`
- `hipfftExecZ2Z`
- `hipfftHandle`
- `hipfftPlan1d`
- `hipfftType::HIPFFT_Z2Z`
- `hipfftXtSetCallback`
- `HIPFFT_CB_LD_COMPLEX_DOUBLE`
- `HIPFFT_FORWARD`

### HIP runtime

- `HIP_SYMBOL`
- `hipCmul`
- `hipFree`
- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyFromSymbol`
- `hipMemcpyHostToDevice`
- `make_hipDoubleComplex`
