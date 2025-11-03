# rocCV Custom Crop

## Description

This example illustrates the use of the `rocCV` library for custom cropping operations, which extract rectangular regions from specific coordinates within images.

The custom crop operation allows precise specification of crop regions by defining the top-left corner coordinates and dimensions, enabling applications like region of interest extraction, focused image analysis, and selective content preservation. Unlike center crop, custom crop provides complete control over the crop position.

## Application flow

1. Set up command line arguments including input/output file paths, device selection, and crop rectangle parameters (crop_x, crop_y, crop_width, crop_height).
2. Parse command line arguments and validate crop parameters.
3. Set up the compute device (GPU or CPU) and create HIP stream if using GPU.
4. Load the input image using OpenCV and validate that it loaded successfully.
5. If no crop parameters are provided, set safe default crop values (centered quadrant of the image).
6. Define the crop rectangle using `Box_t` structure with specified position and dimensions.
7. Create input tensor with NHWC layout and U8 data type for the original image dimensions.
8. Create output tensor with NHWC layout and U8 data type for the cropped image dimensions.
9. Copy input image data from host to device memory (or directly to host memory for CPU mode).
10. Execute the custom crop operation on the specified device using the defined crop rectangle.
11. Copy the cropped result from device to host memory (or directly from host memory for CPU mode).
12. Save the output image to disk using OpenCV.
13. Display processing results including original and cropped image dimensions and crop coordinates.
14. Clean up resources including HIP stream if GPU mode was used.

## Key APIs and Concepts

- **Custom Crop Operation**: The `CustomCrop` class extracts rectangular regions from user-specified coordinates within images.

- **Crop Rectangle Definition**:
  - `Box_t`: Structure defining rectangular regions with x, y coordinates and width, height.
  - **Position Control**: Precise specification of top-left corner coordinates (x, y).
  - **Dimension Control**: Exact specification of crop width and height.
  - **Flexible Positioning**: Unlike center crop, can extract from any position within the image.

- **Parameter Configuration**:
  - **crop_x**: X coordinate of the top-left corner of the crop rectangle.
  - **crop_y**: Y coordinate of the top-left corner of the crop rectangle.
  - **crop_width**: Width of the crop rectangle.
  - **crop_height**: Height of the crop rectangle.
  - **Safe Defaults**: If no parameters are provided, defaults to center quadrant extraction.

- **Validation and Error Handling**:
  - **Parameter Validation**: Ensures crop dimensions are positive values.
  - **Boundary Checking**: Crop region must fit within source image boundaries.
  - **Default Generation**: Safe fallback parameters when user input is missing.

- **Tensor Operations**:
  - **Dynamic Sizing**: Output tensor dimensions are calculated based on crop rectangle parameters.
  - **Memory Management**: Automatic allocation and deallocation of input and output tensors.
  - **Stream Processing**: GPU operations use HIP streams for efficient execution.

- **Device Compatibility**:
  - **GPU Acceleration**: Leverages AMD GPU hardware for parallel processing.
  - **CPU Fallback**: Processes images on CPU when GPU is not available.
  - **Unified Interface**: Same API calls work for both GPU and CPU execution modes.

- **Memory Management**:
  - **Host-to-Device Transfers**: Input image data is efficiently copied to GPU memory.
  - **Device-to-Host Transfers**: Result data is copied back to host memory for saving.
  - **Asynchronous Operations**: Memory transfers use asynchronous operations for better performance.

- **Use Cases**:
  - **Region of Interest**: Extracting specific areas of interest from larger images.
  - **Content Selection**: Selecting particular content while discarding surrounding areas.
  - **Preprocessing**: Preparing image regions for further analysis or processing.
  - **Selective Analysis**: Focusing processing on specific image regions.

## Demonstrated API Calls

### rocCV

- `roccv::CustomCrop::operator()`
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
- `roccv::Box_t`
- `roccv::CustomCrop`
- `roccv::TensorDataStrided`
- `roccv::eTensorLayout::TENSOR_LAYOUT_NHWC`
- `roccv::eDataType::DATA_TYPE_U8`
- `roccv::eDeviceType::GPU`
- `roccv::eDeviceType::CPU`
