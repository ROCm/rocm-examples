# rocCV Gamma Contrast

## Description

This example illustrates the use of the `rocCV` library for gamma correction operations, which adjust the brightness and contrast of images by applying non-linear transformations to pixel values.

The gamma correction operation applies the formula: `output = 255 * (input/255)^(1/gamma)`, where gamma values greater than 1.0 darken the image and values less than 1.0 brighten it. This is commonly used to adjust image brightness while preserving relative contrast relationships.

## Application flow

1. Set up command line arguments including input/output file paths, device selection, and gamma value parameter.
2. Parse command line arguments and validate the gamma parameter.
3. Set up the compute device (GPU or CPU) and create HIP stream if using GPU.
4. Load the input image using OpenCV and validate that it loaded successfully.
5. Create input and output tensors with NHWC layout and U8 data type for the image dimensions.
6. Copy input image data from host to device memory (or directly to host memory for CPU mode).
7. Execute the gamma correction operation with the specified gamma value on the chosen device.
8. Copy the gamma-corrected result from device to host memory (or directly from host memory for CPU mode).
9. Save the output image to disk using OpenCV.
10. Display processing results including image dimensions and gamma value used.
11. Clean up resources including HIP stream if GPU mode was used.

## Key APIs and Concepts

- **Gamma Correction Operation**: The `GammaContrast` class applies non-linear brightness adjustments to images using gamma transformation.

- **Gamma Parameter**:
  - **Gamma Value**: Controls the shape of the correction curve.
    - `gamma > 1.0`: Darkens the image (compresses bright regions).
    - `gamma = 1.0`: No change (linear transformation).
    - `gamma < 1.0`: Brightens the image (expands dark regions).
  - **Typical Values**: Common gamma values include 2.2 (standard display gamma), 1.8 (Macintosh), and custom values for artistic effects.

- **Mathematical Formula**:
  - Normalized input: `input_norm = input / 255.0`
  - Gamma correction: `output_norm = input_norm^(1/gamma)`
  - Denormalized output: `output = output_norm * 255.0`
  - This preserves the 0-255 range while applying non-linear transformation.

- **Batch Processing**:
  - Batch size parameter allows processing multiple images simultaneously.
  - Each image in the batch can have different dimensions but should have the same number of channels.
  - Efficient memory allocation for batch operations.

- **Tensor Configuration**:
  - **NHWC Layout**: Batch-Height-Width-Channels layout optimized for image processing.
  - **U8 Data Type**: 8-bit unsigned integers representing pixel values (0-255 range).
  - **Automatic Memory Management**: Tensors handle allocation and deallocation automatically.

- **Device Execution**:
  - **GPU Acceleration**: Uses AMD GPU hardware for parallel processing of image data.
  - **CPU Fallback**: Processes images on CPU when GPU is not available.
  - **Unified API**: Same function calls work for both GPU and CPU execution modes.

- **Memory Operations**:
  - **Asynchronous Transfers**: Uses `hipMemcpyAsync()` for efficient GPU memory operations.
  - **Stream Synchronization**: Ensures operations complete before accessing results.
  - **Host-Device Interface**: Seamless data movement between CPU and GPU memory spaces.

## Demonstrated API Calls

### rocCV

- `roccv::GammaContrast::operator()`
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
- `roccv::GammaContrast`
- `roccv::TensorDataStrided`
- `eTensorLayout::TENSOR_LAYOUT_NHWC`
- `eDataType::DATA_TYPE_U8`
- `eDeviceType::GPU`
- `eDeviceType::CPU`
