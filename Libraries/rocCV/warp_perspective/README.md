# rocCV Warp Perspective

## Description

This example illustrates the use of the `rocCV` library for perspective transformation operations, which apply geometric transformations to images based on 3x3 perspective transformation matrices.

The warp perspective operation maps image coordinates from one perspective to another, enabling applications like correcting perspective distortion, creating bird's-eye views, and applying artistic perspective effects.

## Application flow

1. Set up command line arguments including input/output file paths, device selection, interpolation type, and border mode.
2. Parse and validate command line arguments, including interpolation and border mode parameters.
3. Set up the compute device (GPU) and create HIP stream.
4. Load the input image using OpenCV.
5. Create input and output tensors with NHWC layout and U8 data type for the image dimensions.
6. Copy input image data from host to device memory using asynchronous HIP memory copy.
7. Define the perspective transformation matrix that maps source coordinates to destination coordinates.
8. Configure the warp perspective operation with the transformation matrix, interpolation type, and border handling mode.
9. Execute the warp perspective operation on the GPU using the specified parameters.
10. Copy the transformed result from device to host memory using asynchronous HIP memory copy.
11. Synchronize the HIP stream to ensure all operations are completed.
12. Save the output image to disk using OpenCV.
13. Display processing results including image dimensions and transformation parameters.
14. Clean up resources including HIP stream.

## Key APIs and Concepts

- **Perspective Transformation**: The `WarpPerspective` class applies 3D perspective transformations to images using 3x3 transformation matrices.

- **Transformation Matrix**:
  - `PerspectiveTransform`: 3x3 matrix that defines the mapping between source and destination coordinates.
  - The matrix transforms homogeneous coordinates: `[x', y', w']^T = M * [x, y, 1]^T`
  - Final coordinates are obtained by dividing by w': `[x_dst, y_dst] = [x'/w', y'/w']`

- **Interpolation Methods**:
  - `eInterpolationType::INTERP_NEAREST`: Nearest neighbor interpolation (fastest, lowest quality).
  - `eInterpolationType::INTERP_LINEAR`: Bilinear interpolation (balanced speed and quality).
  - `eInterpolationType::INTERP_CUBIC`: Bicubic interpolation (slowest, highest quality).

- **Border Handling**:
  - `eBorderType::BORDER_CONSTANT`: Pixels outside the image are filled with a constant value.
  - `eBorderType::BORDER_REPLICATE`: Edge pixels are replicated to fill outside areas.
  - `eBorderType::BORDER_REFLECT`: Image is reflected at the border to fill outside areas.
  - `eBorderType::BORDER_WRAP`: Image wraps around at the borders.

- **Color Handling**:
  - `make_float4(r, g, b, a)`: Creates RGBA color values for border fill operations.
  - Border color is specified as separate RGBA components with floating-point precision.

- **Memory Management**:
  - Asynchronous memory transfers using `hipMemcpyAsync()` for improved performance.
  - Stream synchronization using `hipStreamSynchronize()` to ensure operation completion.
  - Tensor data access through `exportData<TensorDataStrided>()` for memory operations.

- **Tensor Configuration**:
  - NHWC tensor layout optimized for image processing operations.
  - U8 data type for standard 8-bit image representation.
  - Automatic memory allocation and management for input and output tensors.

## Demonstrated API Calls

### rocCV

- `roccv::WarpPerspective::operator()`
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
- `roccv::eInterpolationType`
- `roccv::eBorderType`
- `roccv::WarpPerspective`
- `roccv::PerspectiveTransform`
- `roccv::TensorDataStrided`
- `roccv::eTensorLayout::TENSOR_LAYOUT_NHWC`
- `roccv::eDataType::DATA_TYPE_U8`
- `roccv::eInterpolationType::INTERP_NEAREST`
- `roccv::eInterpolationType::INTERP_LINEAR`
- `roccv::eInterpolationType::INTERP_CUBIC`
- `roccv::eBorderType::BORDER_CONSTANT`
- `roccv::eBorderType::BORDER_REPLICATE`
- `roccv::eBorderType::BORDER_REFLECT`
- `roccv::eBorderType::BORDER_WRAP`
