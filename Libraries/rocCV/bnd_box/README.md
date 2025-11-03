# rocCV Bounding Box

## Description

This example illustrates the use of the `rocCV` library for drawing bounding boxes on images, which are rectangular annotations used to highlight regions of interest in computer vision applications.

The bounding box operation draws rectangles with customizable border and fill colors on images, supporting multiple boxes per image and batch processing. This is commonly used for object detection results, region of interest marking, and annotation visualization.

## Application flow

1. Set up command line arguments including input/output file paths, device selection, and optional bounding box file path.
2. Parse command line arguments and check for bounding box file specification.
3. Set up the compute device (GPU or CPU) and create HIP stream if using GPU.
4. Load the input image using OpenCV.
5. If bounding box file is provided, parse the file to extract box specifications; otherwise use default box configurations.
6. Parse bounding box parameters including position, dimensions, border thickness, and RGBA colors for border and fill.
7. Create input and output tensors with NHWC layout and U8 data type for the image dimensions.
8. Copy input image data from host to device memory (or directly to host memory for CPU mode).
9. Create `BndBoxes` object from parsed bounding box vector for batch processing.
10. Execute the bounding box drawing operation on the specified device.
11. Copy the result image from device to host memory (or directly from host memory for CPU mode).
12. Save the output image to disk using OpenCV.
13. Display processing results including image dimensions and number of bounding boxes processed.
14. Clean up resources including HIP stream if GPU mode was used.

## Key APIs and Concepts

- **Bounding Box Operation**: The `BndBox` class draws rectangular annotations on images with customizable appearance parameters.

- **Bounding Box Structure**:
  - `BndBox_t`: Structure containing box position, dimensions, and appearance settings.
  - `Box_t`: Rectangle definition with x, y coordinates, width, and height.
  - `Color_t`: RGBA color specification with individual components for red, green, blue, and alpha.

- **Box Parameters**:
  - **Position**: (x, y) coordinates of the top-left corner of the box.
  - **Dimensions**: Width and height of the rectangular box.
  - **Border Thickness**: Line thickness for the box outline (negative values indicate filled boxes).
  - **Border Color**: RGBA color for the box outline.
  - **Fill Color**: RGBA color for filling the box interior.

- **Batch Processing**:
  - `BndBoxes`: Container class for managing multiple bounding boxes across multiple images.
  - Supports batch processing with different numbers of boxes per image.
  - Efficient memory management for large numbers of boxes.

- **File Format**:
  - First line: Number of images in the batch.
  - Second line for each image: Number of boxes for that image.
  - For each box: 13 lines containing position, dimensions, thickness, and color information.

- **Color Specification**:
  - RGBA components with 8-bit precision (0-255 range).
  - Alpha channel controls transparency (255 = opaque, 0 = fully transparent).
  - Separate colors for border and fill allow for flexible styling.

- **Default Box Generation**:
  - When no box file is provided, generates default boxes based on image dimensions.
  - Creates multiple boxes with different styles (borders, fills, partial fills).
  - Demonstrates various box positioning and styling capabilities.

- **Memory Management**:
  - Vector-based storage for box specifications before GPU transfer.
  - Efficient packing of box data for GPU processing.
  - Stream-based operations for optimal performance.

## Demonstrated API Calls

### rocCV

- `roccv::BndBox::operator()`
- `roccv::BndBoxes::BndBoxes()`
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
- `roccv::BndBox`
- `roccv::BndBoxes`
- `roccv::BndBox_t`
- `roccv::Box_t`
- `roccv::Color_t`
- `roccv::TensorDataStrided`
- `roccv::eTensorLayout::TENSOR_LAYOUT_NHWC`
- `roccv::eDataType::DATA_TYPE_U8`
- `roccv::eDeviceType::GPU`
- `roccv::eDeviceType::CPU`
