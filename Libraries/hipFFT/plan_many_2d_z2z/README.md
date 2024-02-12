# hipFFT Many 2-dimensional Complex to Complex FFT Plan Example
## Description

This example showcases how to execute a batched 2-dimensional complex-to-complex fast Fourier transform (FFT) with advanced data layout on the GPU.

### Application flow

1. Define parameters for the batch of matrices.
2. Allocate, initialize and print the host input data.
3. Create the FFT plan.
4. Allocate device memory and copy host input data to device.
5. Execute the plan.
6. Copy output data from device to host and print it.
7. Destroy plan and free device memory.

## Key APIs and Concepts

### hipFFT

- The `hipfftPlanMany(...)` can be used for 1,2 and 3D transforms. This is done by taking an array of sizes and strides instead of a value. Also the number of dimensions is a parameter.
- The `hipfftHandle` needs to be created with `hipfftCreate(...)` before use and destroyed with `hipfftDestroy(...)` after use.

## Used API surface

### hipFFT

- `hipfftDestroy`
- `hipfftDoubleComplex`
- `hipfftExecZ2Z`
- `HIPFFT_FORWARD`
- `hipfftHandle`
- `hipfftPlanMany`
- `HIPFFT_Z2Z`

### HIP runtime

- `hipFree`
- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
