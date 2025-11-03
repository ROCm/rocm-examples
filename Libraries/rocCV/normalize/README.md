# rocCV Normalize

## Description

This example illustrates the use of the `rocCV` library for image normalization operations, which apply scaling and shifting transformations to pixel values across different channels.

The normalize operation transforms pixel values using the formula: `output = (input - shift) * scale * global_scale + global_shift`. This is commonly used for preprocessing images in machine learning pipelines and computer vision applications.

## Application flow

1. Set up command line arguments including input/output file paths, device selection, and normalization parameters (global_scale, global_shift, epsilon).
2. Parse command line arguments and check for optional base and scale parameter files.
3. Set up the compute device (GPU or CPU) and create HIP stream if using GPU.
4. Load the input image using OpenCV.
5. If base parameter file is provided, parse the file to extract per-channel shift values; otherwise use default values.
6. If scale parameter file is provided, parse the file to extract per-channel scale values; otherwise use default values.
7. Create scale tensor with the extracted scale parameters and appropriate dimensions.
8. Create base tensor with the extracted shift parameters and appropriate dimensions.
9. Create input/output tensors with NHWC layout and U8 data type for the image dimensions.
10. Copy input image data, scale parameters, and base parameters from host to device memory (or directly to host memory for CPU mode).
11. Configure normalization flags including whether scale parameters represent standard deviation.
12. Execute the normalize operation with specified parameters on the chosen device.
13. Copy the normalized result from device to host memory (or directly from host memory for CPU mode).
14. Save the output image to disk using OpenCV.
15. Display processing results including image dimensions and normalization parameters.
16. Clean up resources including HIP stream if GPU mode was used.

## Key APIs and Concepts

- **Normalization Operation**: The `Normalize` class applies per-channel scaling and shifting operations to image tensors with support for global scaling factors.

- **Parameter Configuration**:
  - **Global Scale**: Multiplicative factor applied to all channels after per-channel scaling.
  - **Global Shift**: Additive factor applied to all channels after scaling.
  - **Base Parameters**: Per-channel shift values (mean subtraction) for each color channel.
  - **Scale Parameters**: Per-channel scale values (standard division or standard deviation) for each color channel.
  - **Epsilon**: Small value added to prevent division by zero when dealing with standard deviation.

- **Parameter File Format**:
  - First line: Number of images in the batch.
  - Second line: Scalar indicator (1 for scalar parameters, 0 for per-pixel parameters).
  - Following lines: Individual parameter values for each channel (typically R, G, B).

- **Normalization Flags**:
  - `ROCCV_NORMALIZE_SCALE_IS_STDDEV`: Flag indicating that scale parameters represent standard deviation values rather than simple scaling factors.

- **Image Format Handling**:
  - `ImageFormat`: Specifies the format of image tensors (e.g., `FMT_RGBf32` for RGB floating-point format).
  - `Size2D`: Structure representing 2D dimensions (width, height) for tensor shapes.

- **Tensor Operations**:
  - Tensor constructors with batch size, dimensions, and format specifications.
  - Memory management for parameter tensors and image data.
  - Stream-based operations for GPU processing.

- **Memory Management**:
  - Host-to-device and device-to-host memory transfers for images and parameters.
  - Asynchronous memory copy operations using HIP streams.
  - Proper synchronization to ensure operation completion.

## Demonstrated API Calls

### rocCV

- `roccv::Normalize::operator()`
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
- `roccv::ImageFormat`
- `roccv::Size2D`
- `roccv::Normalize`
- `roccv::TensorDataStrided`
- `roccv::eTensorLayout::TENSOR_LAYOUT_NHWC`
- `roccv::eDataType::DATA_TYPE_U8`
- `roccv::eDeviceType::GPU`
- `roccv::eDeviceType::CPU`
- `ROCCV_NORMALIZE_SCALE_IS_STDDEV`
- `roccv::FMT_RGBf32`
