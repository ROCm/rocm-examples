# rocCV Crop and Resize

## Description

This example illustrates the use of the `rocCV` library for a combined crop and resize pipeline, which demonstrates batch processing and multi-operator workflows for image preprocessing.

The crop and resize pipeline performs sequential operations: first extracting a central crop region from input images, then resizing the cropped result to a target resolution. This is commonly used in machine learning preprocessing pipelines to standardize image dimensions while preserving important content.

## Application flow

1. Set up command line arguments including input image directory path and batch size parameter.
2. Parse command line arguments and validate the input path and batch size.
3. Handle both single file and directory inputs, collecting JPEG files for batch processing.
4. Validate that all images in a directory have consistent dimensions for batch processing.
5. Load the first image to determine dimensions dynamically and convert BGR to RGB format.
6. Set up the compute device (GPU) and create HIP stream for asynchronous operations.
7. Calculate the largest square crop that can be extracted from the center of images.
8. Create tensors for input, crop, and resize stages with appropriate dimensions and RGB8 format.
9. Allocate GPU memory for the input batch tensor using hipMallocAsync with proper stride calculations.
10. Load and copy all images for batch processing to GPU memory at appropriate offsets.
11. Define the crop rectangle parameters for the center crop operation.
12. Initialize the CustomCrop and Resize operators for the pipeline.
13. Execute the center crop operation to extract the central square region.
14. Execute the resize operation with linear interpolation to scale to 320x320 resolution.
15. Profile the operations (if enabled) using HIP events for timing measurement.
16. Copy the final resized results from device to host and write individual BMP files.
17. Clean up GPU resources including the HIP stream.

## Key APIs and Concepts

- **Pipeline Processing**: Demonstrates a multi-stage processing pipeline with crop followed by resize operations.

- **Batch Processing**:
  - **Dynamic Batch Size**: Supports configurable batch sizes for processing multiple images simultaneously.
  - **Directory Handling**: Automatically collects JPEG files from input directories.
  - **Dimension Validation**: Ensures all images in a batch have consistent dimensions.
  - **Memory Offset Management**: Calculates proper memory offsets for batch data storage.

- **Tensor Configuration**:
  - **Dynamic Sizing**: Tensors are created based on actual image dimensions rather than fixed values.
  - **RGB8 Format**: Uses 8-bit RGB format for optimal GPU processing.
  - **Stride Calculations**: Proper stride calculations for efficient memory access patterns.
  - **Tensor Requirements**: Calculates memory requirements for tensor allocation.

- **Crop Operation**:
  - **Center Cropping**: Automatically calculates the largest square crop from image center.
  - **CustomCrop Operator**: Uses the `CustomCrop` class for precise rectangular extraction.
  - **Box_t Structure**: Defines crop regions with x, y coordinates and width, height.
  - **Batch Consistency**: All images in batch are cropped using the same parameters.

- **Resize Operation**:
  - **Linear Interpolation**: Uses `INTERP_TYPE_LINEAR` for balanced quality and performance.
  - **Target Resolution**: Resizes all images to consistent 320x320 square dimensions.
  - **Resize Operator**: Uses the `Resize` class for efficient scaling operations.

- **Memory Management**:
  - **Buffer Allocation**: Manual GPU memory allocation using `hipMallocAsync`.
  - **Tensor Wrapping**: Uses `TensorWrapData` to wrap existing GPU memory buffers.
  - **Asynchronous Operations**: All memory transfers use async operations for performance.
  - **Proper Cleanup**: Ensures all allocated GPU memory is freed.

- **Color Format Handling**:
  - **BGR to RGB Conversion**: Converts OpenCV's default BGR format to RGB for processing.
  - **RGB Output**: Maintains RGB format throughout the pipeline.
  - **File Format Conversion**: Converts RGB back to BGR for BMP file output.

- **Performance Profiling**:
  - **Event Timing**: Uses HIP events to measure kernel execution time.
  - **Conditional Compilation**: Profiling code is included based on PROFILE_SAMPLE macro.
  - **Pipeline Timing**: Measures the combined time for crop and resize operations.

## Demonstrated API Calls

### rocCV

- `roccv::CustomCrop::operator()`
- `roccv::Resize::operator()`
- `roccv::Tensor::CalcRequirements`
- `roccv::TensorWrapData`
- `roccv::Tensor::operator()`
- `roccv::Tensor::layout`
- `roccv::Tensor::shape`

### HIP runtime

- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`
- `hipMallocAsync`
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
- `roccv::TensorDataStrided`
- `roccv::ImageFormat`
- `roccv::CustomCrop`
- `roccv::Resize`
- `roccv::Box_t`
- `roccv::FMT_RGB8`
- `INTERP_TYPE_LINEAR`
