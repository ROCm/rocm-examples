# rocCV Bilateral Filter

## Description

This example illustrates the use of the `rocCV` library for bilateral filtering, which is a non-linear, edge-preserving, and noise-reducing smoothing filter for images.

The bilateral filter applies a weighted average to each pixel, where the weights depend on both the spatial distance and the intensity difference between pixels. This preserves edges while smoothing noise in homogeneous regions.

## Application flow

1. Set up command line arguments including input/output file paths, device selection, and filter parameters (diameter, sigma_color, sigma_space).
2. Parse and validate command line arguments, including border mode and color parameters.
3. Set up the compute device (GPU or CPU) and create HIP stream if using GPU.
4. Load the input image using OpenCV.
5. Create input and output tensors with NHWC layout and U8 data type for the image dimensions.
6. Copy input image data from host to device memory (or directly to host memory for CPU mode).
7. Configure bilateral filter parameters including filtering area diameter, spatial and color sigma values, border mode, and border color.
8. Execute the bilateral filter operation on the specified device.
9. Copy the filtered result from device to host memory (or directly from host memory for CPU mode).
10. Save the output image to disk using OpenCV.
11. Display processing results including image dimensions and filter parameters.
12. Clean up resources including HIP stream if GPU mode was used.

## Key APIs and Concepts

- **rocCV Tensor Operations**: The rocCV library uses tensor objects to store and manipulate image data efficiently on GPU or CPU memory.

- **Tensor Configuration**:
  - `TensorShape`: Defines the dimensions and layout of the tensor, typically using NHWC (Batch, Height, Width, Channels) layout for images.
  - `DataType`: Specifies the data type of tensor elements, commonly `DATA_TYPE_U8` for 8-bit unsigned integers representing pixel values.
  - `eDeviceType`: Determines whether operations run on GPU (`eDeviceType::GPU`) or CPU (`eDeviceType::CPU`).

- **Bilateral Filter Operation**:
  - `BilateralFilter`: The main operator class that performs bilateral filtering on input tensors.
  - `Diameter`: Specifies the diameter of the pixel neighborhood used for filtering (odd number recommended).
  - `Sigma Space`: Controls the spatial Gaussian function - larger values mean farther pixels influence each other.
  - `Sigma Color`: Controls the range Gaussian function - larger values mean more colors within the range are mixed together.
  - `Border Handling`: Different modes for handling edges and borders of the image during filtering.

- **Border Modes**:
  - `eBorderType::BORDER_CONSTANT`: Border is filled with a constant color value.
  - `eBorderType::BORDER_REPLICATE`: Edge pixels are replicated to create the border.
  - `eBorderType::BORDER_REFLECT`: Border is created by reflecting pixels at the edge.
  - `eBorderType::BORDER_WRAP`: Border is created by wrapping around to the opposite side.

- **Memory Management**:
  - `exportData<TensorDataStrided>()`: Provides access to the underlying tensor data buffer for memory operations.
  - `hipMemcpyAsync()`: Asynchronous memory copy operations between host and device memory.
  - `hipStreamSynchronize()`: Ensures all operations in a stream are completed before proceeding.

- **Color Components**:
  - `float4`: Structure for storing RGBA color values with floating-point precision for border color specification.

## Demonstrated API Calls

### rocCV

- `roccv::BilateralFilter::operator()`
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
- `roccv::eBorderType`
- `roccv::BilateralFilter`
- `roccv::TensorDataStrided`
- `roccv::eTensorLayout::TENSOR_LAYOUT_NHWC`
- `roccv::eDataType::DATA_TYPE_U8`
- `roccv::eDeviceType::GPU`
- `roccv::eDeviceType::CPU`
