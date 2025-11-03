# rocCV Composite

## Description

This example illustrates the use of the `rocCV` library for image compositing operations, which combine multiple image layers using alpha blending to create composite images.

The composite operation merges foreground and background images using a mask to control pixel-by-pixel blending, enabling applications like overlay effects, layering, and selective image combination. When no input images are provided, the sample generates default images demonstrating the compositing functionality.

## Application flow

1. Set up command line arguments including background, foreground, and mask file paths, device selection, and output file path.
2. Parse command line arguments and validate file paths.
3. Set up the compute device (GPU) and create HIP stream.
4. Generate default images if any of background, foreground, or mask files are not provided:
   - Background: Blue gradient image
   - Foreground: Red circle on white background
   - Mask: Circular gradient with soft edges
5. Load input images using OpenCV for any provided file paths.
6. Create separate tensors for background, foreground, and mask with appropriate dimensions and NHWC layout.
7. Copy input image data from host to device memory using asynchronous HIP memory copies.
8. Create HIP events for performance timing and recording.
9. Execute the composite operation, blending foreground onto background using the mask.
10. Record performance timing and synchronize to measure execution time.
11. Copy the composited result from device to host memory using asynchronous HIP memory copy.
12. Synchronize the HIP stream to ensure all operations are completed.
13. Save the output image to disk using OpenCV.
14. Display processing results including image sources, dimensions, and execution time.
15. Clean up resources including HIP stream and events.

## Key APIs and Concepts

- **Composite Operation**: The `Composite` class performs alpha blending of foreground and background images using mask-based pixel selection.

- **Alpha Blending Formula**:
  - Final pixel = `foreground * mask + background * (1 - mask)`
  - Mask values range from 0 (fully transparent) to 255 (fully opaque)
  - Enables smooth transitions and layering effects

- **Image Generation**:
  - **Background**: Blue gradient image using mathematical coloring based on pixel coordinates
  - **Foreground**: Red-filled circle on white background using OpenCV drawing functions
  - **Mask**: Circular gradient with Gaussian blur for soft, natural-looking edges

- **Performance Measurement**:
  - `hipEventCreate()`: Creates timing events for measuring kernel execution time
  - `hipEventRecord()`: Records timestamps before and after kernel execution
  - `hipEventElapsedTime()`: Calculates elapsed time between events in milliseconds
  - `hipEventSynchronize()`: Ensures events are completed before reading timing data

- **Memory Management**:
  - **Asynchronous Transfers**: All memory copies use `hipMemcpyAsync()` for optimal performance
  - **Stream Synchronization**: `hipStreamSynchronize()` ensures completion before accessing results
  - **Event Synchronization**: `hipEventSynchronize()` ensures timing events are completed

- **Tensor Operations**:
  - **Separate Tensors**: Individual tensors for background, foreground, and mask layers
  - **Consistent Layout**: All tensors use NHWC layout for optimal GPU processing
  - **Flexible Dimensions**: Supports different image sizes with appropriate tensor creation

- **Color Handling**:
  - **BGR to RGB Conversion**: OpenCV loads images in BGR format; conversion may be needed for processing
  - **Mask Processing**: Single-channel grayscale image for alpha values
  - **Output Format**: Maintains 3-channel color output for compatibility

- **Batch Processing Support**:
  - Designed for single images but can be extended to batch processing
  - Efficient memory allocation patterns for multiple tensors
  - Stream-based operations for optimal GPU utilization

## Demonstrated API Calls

### rocCV

- `roccv::Composite::operator()`
- `roccv::Tensor::Tensor()`
- `roccv::TensorShape::TensorShape()`
- `roccv::Tensor::exportData`

### HIP runtime

- `hipSetDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`
- `hipMemcpyAsync`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
- `hipEventCreate`
- `hipEventDestroy`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipEventElapsedTime`

### Data Types and Enums

- `roccv::Tensor`
- `roccv::TensorShape`
- `roccv::TensorLayout`
- `roccv::DataType`
- `roccv::Composite`
- `roccv::TensorDataStrided`
- `roccv::eTensorLayout::TENSOR_LAYOUT_NHWC`
- `roccv::eDataType::DATA_TYPE_U8`
