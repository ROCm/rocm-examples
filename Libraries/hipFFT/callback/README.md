# hipFFT Callback Example

## Description

This example illustrates the use of hipFFT `callback` functionality. It shows how to use load callbacks, a user-defined callback function that is run to load input from global memory at the start of the transform, with hipFFT.

### Application flow

1. Allocate and initialize the host input data and filter.
2. Allocate device memory and copy host input data and filter from host to device.
3. Compile the JIT load callback from source code.
4. Copy callback data from host to device.
5. Allocate and initialize callback data on host.
6. Allocate a new plan.
7. Set the callback and callback data on the plan.
8. Initialize the plan as a 1D FFT.
9. Execute FFT plan which multiplies each element by filter element and scales.
10. Copy the results from device to host and print it.
11. Destroy plan and free device memory.

## Key APIs and Concepts

### hipFFT

- The `hipfftHandle` needs to be created with `hipfftCreate(...)` before use and destroyed with `hipfftDestroy(...)` after use.
- This example compiles and sets a load callback on the `hipfftHandle` with `hipfftXtSetJITCallback` prior to initializing the handle.  The compiled function and callback data are passed to hipFFT at this point.

## Used API surface

### hipFFT

- `hipfftCreate`
- `hipfftDestroy`
- `hipfftExecZ2Z`
- `hipfftPlan1d`
- `hipfftXtSetJITCallback`

### HIP runtime

- `hipCmul`
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `make_hipDoubleComplex`
