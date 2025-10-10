# RPP Contrast

## Description

This example illustrates the use of the `RPP` library for adjusting image contrast. The contrast operation applies a transformation that scales pixel values around a center point using the formula:

$output = factor \cdot (input - center) + center$

where $factor$ controls the contrast intensity and $center$ is the pivot point (typically 128 for U8 images).

## Application flow

1. Parse command-line arguments for input/output paths, bit depth, layout type, contrast factor, and center point.
2. Load image names from the input folder.
3. Read images using OpenCV to determine maximum dimensions.
4. Initialize RPP tensor descriptors for source and destination with appropriate layouts and data types.
5. Set up ROI (Region of Interest) tensors for batch processing.
6. Calculate buffer sizes and allocate host memory.
7. Read image batch into host memory and convert layout if needed (PKD3 to PLN3).
8. Convert input data to the specified bit depth (U8, F16, F32, or I8).
9. Allocate device memory and copy input data from host to device.
10. Create RPP handle with HIP backend and configure stream.
11. Set up contrast factor and center parameter tensors for batch processing.
12. Execute contrast adjustment operation on GPU.
13. Copy results from device to host memory.
14. Convert output data back to U8 format for visualization.
15. Write processed images to the output folder using OpenCV.
16. Clean up allocated resources and destroy RPP handle.

## Key APIs and Concepts

- **RPP Initialization**: The RPP library is initialized by creating a handle with `rppCreate()` and released with `rppDestroy()`. The handle is configured for the HIP backend (`RPP_HIP_BACKEND`) and associated with a HIP stream.

- **Tensor Descriptors**:
  - `RpptDesc`: Structure defining tensor properties including layout, data type, dimensions (n, c, h, w), strides, and offset.
  - Layout types support both packed (`NHWC`) and planar (`NCHW`) formats for 3-channel images, as well as single-channel planar format.
  - Data types include `U8`, `F16`, `F32`, and `I8` for different precision requirements.
  - Strides define memory layout for efficient access patterns.

- **ROI Management**:
  - `RpptROI`: Defines regions of interest for processing, supporting both XYWH (x, y, width, height) and LTRB (left, top, right, bottom) formats.
  - `RpptRoiType`: Specifies the ROI format being used (`XYWH` or `LTRB`).
  - ROI tensors enable batch processing with different regions per image.

- **Contrast Operation**:
  - `rppt_contrast_gpu()`: Executes the contrast adjustment on GPU, taking source and destination descriptors, contrast factor and center parameter tensors, and ROI information.
  - Contrast factor values greater than 1.0 increase contrast, while values less than 1.0 decrease contrast.
  - The center point defines the pivot around which contrast scaling occurs.
  - Both parameters can be specified per image in the batch.

- **Data Type Conversions**:
  - Utility functions convert between U8, F16, F32, and I8 formats.
  - Normalization factors (1/255) convert U8 data to floating-point range [0, 1].
  - Conversions enable mixed-precision workflows and output visualization.

- **Layout Conversions**:
  - `convert_pkd3_to_pln3()`: Converts packed 3-channel (RGBRGBRGB...) to planar (RRR...GGG...BBB...).
  - `convert_pln3_to_pkd3()`: Converts planar back to packed format for output.

- **Key Enumerations**:
  - `RpptLayout`: Specifies tensor layout (`NHWC`, `NCHW`).
  - `RpptDataType`: Defines data precision (`U8`, `F16`, `F32`, `I8`).
  - `RppBackend`: Backend selection (`RPP_HIP_BACKEND`).

## Demonstrated API Calls

### RPP

- `rppCreate`
- `rppDestroy`
- `rppt_contrast_gpu`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipHostMalloc`
- `hipHostFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipStreamCreate`
- `hipStreamDestroy`

### Data Types and Enums

- `rppHandle_t`
- `hipStream_t`
- `RppBackend`
- `RpptDesc`
- `RpptDescPtr`
- `RpptROI`
- `RpptROIPtr`
- `RpptRoiType`
- `RpptLayout`
- `RpptDataType`
- `RpptImagePatch`
- `Rpp8u`
- `Rpp8s`
- `Rpp16f`
- `Rpp32f`
- `Rpp32u`
- `Rpp64u`
- `RPP_HIP_BACKEND`
- `NHWC`
- `NCHW`
- `XYWH`
- `LTRB`
