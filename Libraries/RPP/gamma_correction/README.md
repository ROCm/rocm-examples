# RPP Gamma Correction

## Description

This example illustrates the use of the `RPP` library for applying gamma correction to images. Gamma correction is a nonlinear operation used to adjust image brightness and contrast by applying a power-law transformation:

$output = input^{\gamma}$

where $\gamma$ is the gamma value. Common gamma values include:

- $\gamma < 1.0$: Brightens the image (expands dark regions)
- $\gamma = 1.0$: No change
- $\gamma > 1.0$: Darkens the image (compresses dark regions)

## Application flow

1. Parse command-line arguments for input/output paths, bit depth, layout type, and gamma value.
2. Load image names from the input folder and validate gamma value.
3. Read images using OpenCV to determine maximum dimensions.
4. Initialize RPP tensor descriptors for source and destination with appropriate layouts and data types.
5. Set up ROI (Region of Interest) tensors for batch processing.
6. Calculate buffer sizes and allocate host memory.
7. Read image batch into host memory and convert layout if needed (PKD3 to PLN3).
8. Convert input data to the specified bit depth (U8, F16, F32, or I8).
9. Allocate device memory and copy input data from host to device.
10. Create RPP handle with HIP backend and configure stream.
11. Set up gamma parameter tensor for batch processing.
12. Execute gamma correction operation on GPU.
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

- **Gamma Correction Operation**:
  - `rppt_gamma_correction_gpu()`: Executes the gamma correction on GPU, taking source and destination descriptors, gamma parameter tensor, and ROI information.
  - Gamma values typically range from 0.5 to 2.5 for practical image processing.
  - The operation applies a power-law transformation to adjust tonal distribution.
  - Each image in the batch can have a different gamma value.

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
- `rppt_gamma_correction_gpu`

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
