# hipFFT Complex to Complex FFT Plan  Example

## Description

This example showcases how to execute a 1, 2, and 3-dimensional complex-to-complex fast Fourier
transform (FFT) on the GPU. There are only slight differences in planning and executing FFT on
different dimensional data.

### Application flow

1. Define the various input parameters.
2. Generate the input data on host.
3. Allocate memory on device for the input and output.
4. Copy the input data from host to device.
5. Create the FFT plan.
6. Execute the plan.
7. Allocate memory on host for the output.
8. Copy output data from device to host.
9. Print the output
10. Clean up.

## Key APIs and Concepts

### hipFFT

- `hipfftPlan[n]d` is used to create a plan for a $n \in \{ 1, 2, 3 \}$-dimensional FFT.
- The `hipfftHandle` needs to be created with `hipfftCreate(...)` before use and destroyed with `hipfftDestroy(...)` after use.

## Used API surface

### hipFFT

- `hipfftCreate`
- `hipfftDestroy`
- `hipfftDoubleComplex`
- `hipfftExecZ2Z`
- `hipfftHandle`
- `hipfftPlan1d`
- `hipfftPlan2d`
- `hipfftPlan3d`
- `hipfftType::HIPFFT_Z2Z`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
