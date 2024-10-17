# hipFFT Mutli GPU Example

## Description

This example showcases how to execute a 2-dimensional complex-to-complex fast Fourier
transform (FFT) with multiple GPUs. Note that the API used is experimental and requires
at least ROCm 6.0.

### Application flow

1. Define the various input parameters.
2. Generate the input data on host.
3. Initialize the FFT plan handle.
4. Set up multi GPU execution.
5. Make the 2D FFT plan.
6. Allocate memory on device.
7. Copy data from host to device.
8. Execute multi GPU FFT from plan.
9. Copy data from device to host.
10. Clean up.

### Command line interface

The application provides the following optional command line arguments:

- `-l` or `--length`. The 3-D FFT size separated by spaces. It default value is `8 8 8`.
- `-d` or `--devices`. The list of devices to use separated by spaces. It default value is `0 1`.

## Key APIs and Concepts

- The `hipfftHandle` needs to be created with `hipfftCreate(...)` before use and destroyed with `hipfftDestroy(...)` after use.
  It can be associated with multiple GPUs.
- `hipfftXtSetGPUs` instructs a plan to use multiple GPUs.
- `hipfhipfftMakePlan2dftPlan2d` is used to create a plan for a 2-dimensional FFT.
- `hipfftXtExecDescriptor` can execute a multi GPU plan.
- Device memory management:
  - `hipfftXtMalloc` allocates device memory for a plan associated with multiple devices.
  - `hipLibXtDesc` holds the handles to device memory on multiple devices.
  - `hipfftXtMemcpy` can copy data between a contiguous host buffer and `hipLibXtDesc`, or between two `hipLibXtDesc`.
  - The memory allocated on device can be freed with `hipfftXtFree`.

## Demonstrated API Calls

### hipFFT

- `HIPFFT_FORWARD`
- `hipfftCreate`
- `hipfftDestroy`
- `hipfftHandle`
- `hipfftMakePlan2d`
- `hipfftSetStream`
- `hipfftType`
  - `HIPFFT_Z2Z`
- `hipfftXtCopyType`
  - `HIPFFT_COPY_DEVICE_TO_HOST`
  - `HIPFFT_COPY_HOST_TO_DEVICE`
- `hipfftXtFree`
- `hipfftXtMalloc`
- `hipfftXtMemcpy`
- `hipfftXtSetGPUs`
- `hipfftXtSubFormat`
  - `HIPFFT_XT_FORMAT_INPUT`
  - `HIPFFT_XT_FORMAT_OUTPUT`
- `hipfftXtExecDescriptor`

### HIP runtime

- `hipGetDeviceCount`
- `hipStream_t`
- `hipStreamCreate`
- `hipStreamDestroy`
