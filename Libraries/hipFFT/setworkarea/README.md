# hipFFT Set Work Area FFT Example

## Description

This example showcases the set work area feature with a simple 1-dimensional real-to-complex fast Fourier transform (FFT) on the GPU.

### Application flow

1. Allocate, initialize and print the input data on host.
2. Allocate memory on device for the input and output.
3. Copy the input data from host to device.
4. Estimate the size for the work buffer.
5. Create a plan without the set auto-allocation flag and use the estimation for work buffer.
6. Allocate and set work buffer on the device.
7. Execute the plan.
8. On the host allocate memory, copy back output data from device to host and print output.
9. Clean up.

## Key APIs and Concepts

### hipFFT
- `hipfftSetAutoAllocation` is used to set the auto-allocation flag, it is in this example set to zero to be able to create our own work area.
- `hipfftEstimate[n]d` is used to estimate the work size for a $n \in \{ 1, 2, 3 \}$-dimensional FFT.
- `hipfftMakePlan[n]d` is used to modify an already existing plan for a $n \in \{ 1, 2, 3 \}$-dimensional FFT.
- `hipfftSetWorkArea` is used to set the work area for a plan after allocating the work area on the device. 

## Used API surface

### hipFFT

- `hipfftComplex`
- `hipfftCreate`
- `hipfftDestroy`
- `hipfftEstimate1d`
- `hipfftExecR2C`
- `hipfftGetSize`
- `hipfftHandle`
- `hipfftMakePlan1d`
- `hipfftReal`
- `hipfftSetAutoAllocation`
- `hipfftSetWorkArea`
- `HIPFFT_R2C`

### HIP runtime

- `hipFree`
- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
