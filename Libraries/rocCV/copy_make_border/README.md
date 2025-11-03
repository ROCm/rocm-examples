# rocCV Copy Make Border

## Description

This example illustrates the use of the `rocCV` library for copy and border operations, which add padding around images using various border handling modes.

The copy make border operation creates a larger output image by adding specified border widths to all sides of the input image, with configurable border colors and handling modes. This is commonly used for padding operations, kernel padding in convolution operations, and ensuring consistent image dimensions.

## Application flow

1. Set up command line arguments including input/output file paths, device selection, border dimensions, color values, and border mode.
2. Parse and validate command line arguments, including border mode values (0-3).
3. Set up the compute device (GPU) and create HIP stream.
4. Load the input image using OpenCV and validate that it loaded successfully.
5. Create input tensor with NHWC layout and U8 data type for original image dimensions.
6. Create output tensor with NHWC layout and U8 data type for expanded dimensions including borders.
7. Copy input image data from host to device memory using asynchronous HIP memory copy.
8. Configure border color from RGBA components and convert to float4 format.
9. Execute the copy make border operation with specified parameters on the GPU.
10. Copy the bordered result from device to host memory using asynchronous HIP memory copy.
11. Synchronize the HIP stream to ensure all operations are completed.
12. Save the output image to disk using OpenCV.
13. Display processing results including original and output image dimensions and border settings.
14. Clean up resources including HIP stream.

## Key APIs and Concepts

- **Copy Make Border Operation**: The `CopyMakeBorder` class adds padding around images with configurable border styles and colors.

- **Border Modes**:
  - `eBorderType::BORDER_CONSTANT`: Border is filled with a constant color value specified by RGBA components.
  - `eBorderType::BORDER_REPLICATE`: Edge pixels are replicated to create the border (extended edges).
  - `eBorderType::BORDER_REFLECT`: Border is created by reflecting pixels at the edge (mirrored edges).
  - `eBorderType::BORDER_WRAP`: Border is created by wrapping around to the opposite side (tiled).

- **Border Configuration**:
  - **Top Border**: Number of pixels to add above the image.
  - **Left Border**: Number of pixels to add to the left of the image.
  - **Symmetric Borders**: Top/bottom and left/right borders use the same values for consistent padding.
  - **Color Specification**: RGBA floating-point values for border colors in constant mode.

- **Tensor Sizing**:
  - **Input Tensor**: Dimensions match original image size.
  - **Output Tensor**: Dimensions = `(height + 2*top) × (width + 2*left)`.
  - **Automatic Calculation**: Output dimensions are computed based on input size and border parameters.

- **Memory Management**:
  - **Asynchronous Operations**: Uses `hipMemcpyAsync()` for efficient GPU memory transfers.
  - **Stream Processing**: All operations are queued and executed efficiently on HIP streams.
  - **Proper Synchronization**: Ensures completion before accessing results.

- **Color Handling**:
  - **RGBA Format**: Border colors specified as separate red, green, blue, and alpha components.
  - **Float4 Structure**: Uses `make_float4()` to create color vectors for GPU processing.
  - **Precision**: Floating-point values provide smooth color transitions.

- **Error Handling**:
  - **Border Mode Validation**: Ensures border mode is within valid range (0-3).
  - **Image Loading**: Validates that input images are loaded successfully.
  - **Output Saving**: Checks that output images are written successfully.

- **Performance Considerations**:
  - **GPU Acceleration**: Leverages AMD GPU hardware for parallel border operations.
  - **Memory Efficiency**: Minimizes unnecessary memory copies and allocations.
  - **Stream Optimization**: Uses asynchronous operations for maximum throughput.

## Demonstrated API Calls

### rocCV

- `roccv::CopyMakeBorder::operator()`
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
- `roccv::eBorderType`
- `roccv::CopyMakeBorder`
- `roccv::TensorDataStrided`
- `roccv::eTensorLayout::TENSOR_LAYOUT_NHWC`
- `roccv::eDataType::DATA_TYPE_U8`
- `roccv::eBorderType::BORDER_CONSTANT`
- `roccv::eBorderType::BORDER_REPLICATE`
- `roccv::eBorderType::BORDER_REFLECT`
- `roccv::eBorderType::BORDER_WRAP`
