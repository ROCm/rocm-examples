# rocCV Center Crop

## Description

This example illustrates the use of the `rocCV` library for center cropping operations, which extract rectangular regions from the center of images while maintaining aspect ratios.

The center crop operation removes pixels from the outer edges of an image while preserving the central region, commonly used for preprocessing images in machine learning pipelines and ensuring consistent dimensions across datasets.

## Application flow

1. Set up command line arguments including input/output file paths, device selection, and crop dimensions (crop_width, crop_height).
2. Parse command line arguments and validate crop parameters.
3. Set up the compute device (GPU or CPU) and create HIP stream if using GPU.
4. Load the input image using OpenCV.
5. Validate crop dimensions against input image size and use safe defaults if no crop parameters are provided.
6. Create input tensor with NHWC layout and U8 data type for the original image dimensions.
7. Create output tensor with NHWC layout and U8 data type for the cropped image dimensions.
8. Copy input image data from host to device memory (or directly to host memory for CPU mode).
9. Define the crop area using `Size2D` structure with specified width and height.
10. Execute the center crop operation on the specified device.
11. Copy the cropped result from device to host memory (or directly from host memory for CPU mode).
12. Save the output image to disk using OpenCV.
13. Display processing results including original and cropped image dimensions.
14. Clean up resources including HIP stream if GPU mode was used.

## Key APIs and Concepts

- **Center Crop Operation**: The `CenterCrop` class extracts rectangular regions from the center of images, automatically calculating the crop position to center the region.

- **Crop Configuration**:
  - `Size2D`: Structure representing 2D dimensions (width, height) for specifying crop area.
  - Automatic centering: The crop region is automatically positioned to be centered in the source image.
  - Safe defaults: If no crop dimensions are specified, the operation defaults to cropping half the image dimensions.

- **Dimension Handling**:
  - Input validation: Crop dimensions are validated against the source image to prevent invalid operations.
  - Output tensor sizing: Output tensor dimensions are automatically calculated based on crop area.
  - Aspect ratio preservation: Can maintain aspect ratios or use custom dimensions as needed.

- **Tensor Operations**:
  - Dynamic tensor creation: Tensors are created with appropriate dimensions for input and output.
  - Memory management: Automatic allocation and deallocation of tensor memory.
  - Stream-based operations: GPU operations use HIP streams for efficient execution.

- **Device Compatibility**:
  - GPU acceleration: Operations can be executed on AMD GPU devices for improved performance.
  - CPU fallback: Operations can also be executed on CPU for systems without GPU support.
  - Unified interface: Same API calls work for both GPU and CPU execution.

- **Memory Management**:
  - Host-to-device transfers: Input image data is efficiently copied to GPU memory.
  - Device-to-host transfers: Result data is copied back to host memory for saving.
  - Asynchronous operations: Memory transfers use asynchronous operations for better performance.

- **Image Format Support**:
  - NHWC layout: Optimized tensor layout for image processing operations.
  - U8 data type: Standard 8-bit unsigned integer representation for pixel values.
  - Multi-channel support: Handles color images with multiple channels automatically.

## Demonstrated API Calls

### rocCV

- `roccv::CenterCrop::operator()`
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

### Data Types and Enums

- `roccv::Tensor`
- `roccv::TensorShape`
- `roccv::TensorLayout`
- `roccv::DataType`
- `roccv::eDeviceType`
- `roccv::Size2D`
- `roccv::CenterCrop`
- `roccv::TensorDataStrided`
- `roccv::eTensorLayout::TENSOR_LAYOUT_NHWC`
- `roccv::eDataType::DATA_TYPE_U8`
- `roccv::eDeviceType::GPU`
- `roccv::eDeviceType::CPU`
