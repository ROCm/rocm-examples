# HIP-Doc Image Convolution Example

## Description

This example demonstrates 2D image convolution using HIP, implementing a box
blur filter on images. The application uses the stb_image library for image
loading and saving, making it easy to work with common image formats like JPEG
and PNG.

For more information on HIP programming and stencil operations, please refer to
the [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/).

### Application flow

1. An input image is loaded from disk using the stb_image library.
2. A convolution mask (box blur filter) is initialized on the host.
3. Device memory is allocated for the input image, output image, and convolution mask.
4. The input image and mask are copied from host to device memory.
5. A 2D grid of thread blocks is configured based on the image dimensions.
6. The convolution kernel is launched on the GPU.
7. Each thread processes one pixel across all color channels:
   - Applies the convolution mask to the neighborhood around the pixel
   - Handles boundary conditions with zero-padding
   - Normalizes pixel values between 0-255
8. The kernel launch is checked for errors and the device is synchronized.
9. The processed output image is copied back from device to host memory.
10. The output image is saved to disk in JPEG format.
11. All device memory is freed.

### Convolution Implementation

The kernel implements 2D convolution with the following features:

- **Parallel Processing**: Each thread processes one pixel location
- **Multi-channel Support**: Handles RGB images by processing each channel independently
- **Boundary Handling**: Uses zero-padding for pixels near image edges
- **Box Blur Filter**: Applies a uniform averaging filter (33x33 default)
- **Normalized Output**: Maintains pixel values in valid 0-255 range

The box blur filter computes the average of all pixels in the mask region, creating a smoothing/blurring effect.

## Key APIs and Concepts

### HIP Runtime APIs

- `hipMalloc`: Allocates device memory
- `hipMemcpy`: Transfers data between host and device
- `hipFree`: Frees device memory
- `hipGetLastError`: Retrieves the last error from a runtime call
- `hipDeviceSynchronize`: Blocks until all device operations complete

### Device Code Features

- `__global__`: Declares a kernel function callable from host
- `blockIdx`, `blockDim`, `threadIdx`: Built-in variables for grid/block indexing
- 2D thread indexing for image processing

### Stencil Pattern

The convolution operation is a classic stencil computation where each output element depends on a neighborhood of input elements. Key characteristics:

- Regular access pattern (structured grid)
- Halo region handling (boundary conditions)
- Data reuse opportunities (same input pixels used by multiple output pixels)

### Image Processing

- Uses stb_image.h for loading images (JPEG, PNG, BMP, etc.)
- Uses stb_image_write.h for saving images
- Processes images in row-major order with interleaved color channels

## Configuration

- Default input: `test.jpg`
- Default output: `test_out.jpg`
- Default mask size: 33x33 (box blur)
- Block size: 16x16 threads
- Command line usage: `./hip_image_convolution [input.jpg] [output.jpg]`

## Performance Considerations

Potential optimizations for this algorithm:

- Use shared memory to cache frequently accessed pixels
- Separate kernels for different color channels to improve memory coalescing
- Use texture memory for automatic caching and filtering
- Implement separable convolution for larger kernels (two 1D passes instead of one 2D pass)

## Demonstrated API calls

### HIP runtime

#### Device symbols

- `blockDim`
- `blockIdx`
- `threadIdx`

#### Host symbols

- `hipDeviceSynchronize`
- `hipFree`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`

### External Libraries

- `stb_image.h`: Image loading (supports JPEG, PNG, BMP, TGA, etc.)
- `stb_image_write.h`: Image saving (JPEG, PNG, BMP, TGA)
