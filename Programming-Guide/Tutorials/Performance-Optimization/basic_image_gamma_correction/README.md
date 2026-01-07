# AMD ROCm Programming Guide: Basic Image Gamma Correction

## Description

This tutorial demonstrates a basic implementation of image gamma
correction using HIP. Gamma correction is a nonlinear operation used to
adjust the luminance of images. The kernel applies the transformation:
`output = (input / 255)^gamma * 255` to each pixel value.

This example showcases fundamental HIP programming concepts including
kernel launches, memory management, and basic parallel computation
patterns.

### Application flow

1. Image data is loaded (represented conceptually in the code).
2. The kernel launch configuration is calculated based on the number of
   pixel values.
3. The `image_gamma` kernel is launched to process the image in
   parallel.
4. Each thread processes one pixel, applying the gamma correction
   formula.
5. The corrected image data is copied back to the host.
6. The processed image is saved (represented conceptually in the code).
7. Device memory is freed.

## Key APIs and Concepts

- `hipMalloc` allocates memory in the global memory of the device. This
  is required since GPU kernels cannot directly access host memory.
- `hipFree` de-allocates device memory to avoid resource leakage.
- `hipMemcpy` transfers bytes between host and device memory. The
  function synchronizes the device with the host.
- `myKernelName<<<gridDim, blockDim>>>(kernelArguments)` launches a
  kernel on the device. The launch is asynchronous.
  - `gridDim` specifies the number of blocks in the kernel grid.
  - `blockDim` specifies the number of threads in each block.
  - Additional arguments are passed to the kernel function.
- Thread indexing: Each thread computes its global index using
  `threadIdx.x + blockIdx.x * blockDim.x` to determine which pixel to
  process.
- Boundary checking: Threads check if their index is within bounds
  before processing.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`

#### Host symbols

- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`

#### Device functions

- `powf`
