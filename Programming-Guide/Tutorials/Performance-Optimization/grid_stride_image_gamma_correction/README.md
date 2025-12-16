# AMD ROCm Programming Guide: Grid Stride Image Gamma Correction

## Description

This tutorial demonstrates image gamma correction using the grid-stride
loop pattern in HIP. The grid-stride loop is a common GPU programming
pattern that allows a kernel to process datasets larger than the number
of available threads by having each thread process multiple elements in
a strided loop.

This example builds upon the basic gamma correction tutorial by
introducing the grid-stride pattern, which provides better scalability
and flexibility in handling varying image sizes.

### Application flow

1. Image data is loaded (represented conceptually in the code).
2. The kernel launch configuration is defined with a fixed grid size.
3. The `image_gamma` kernel is launched with the grid-stride loop
   pattern.
4. Each thread processes multiple pixels using a strided loop, with
   stride equal to the total number of threads
   (`blockDim.x * gridDim.x`).
5. The corrected image data is copied back to the host.
6. The processed image is saved (represented conceptually in the code).

## Key APIs and Concepts

- **Grid-stride loop pattern**: Instead of calculating grid size based
  on data size, a fixed grid size is used and each thread processes
  multiple elements by looping with stride
  `global_size = blockDim.x * gridDim.x`. This pattern:
  - Allows processing arbitrarily large datasets
  - Provides better scalability across different GPU architectures
  - Enables kernel reuse for different data sizes
- `hipMemcpy` transfers data between host and device memory,
  synchronizing the device.
- Thread indexing: Each thread starts at
  `idx = threadIdx.x + blockIdx.x * blockDim.x` and increments by
  `global_size` in each loop iteration.
- The loop condition `idx < num_values` ensures all pixels are
  processed while preventing out-of-bounds access.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`

#### Host symbols

- `hipMemcpy`
- `hipMemcpyDeviceToHost`

#### Device functions

- `powf`
