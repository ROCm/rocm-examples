# MIVisionX Canny Edge Detection

> [!WARNING]
> MIVisionX will stop being shipped with OpenCV support in future releases, this example requires MIVisionX built with OpenCV support. For more details on how to build MIVisionX, please see https://github.com/ROCm/MIVisionX and https://github.com/ROCm/MIVisionX/pull/1575

## Description

This example demonstrates Canny edge detection using the OpenVX framework. The example processes images or live camera feeds to detect edges using the Canny algorithm, which involves color space conversion, channel extraction, and edge detection with configurable thresholds.

The Canny edge detection pipeline performs:

- RGB to YUV color space conversion
- Luma (Y) channel extraction
- Canny edge detection with hysteresis thresholding

## Application flow

1. Parse command-line arguments for input source and image dimensions.
2. Create OpenVX context and register log callback.
3. Create OpenVX graph for the processing pipeline.
4. Allocate images for input (RGB), intermediate processing (YUV, luma), and output (edge map).
5. Create and configure threshold object with lower and upper hysteresis values.
6. Build processing graph with three nodes:
   - Color conversion node (RGB to YUV)
   - Channel extraction node (extract Y channel)
   - Canny edge detector node
7. Verify the graph to ensure all connections are valid.
8. Process input (either from image file or live camera feed):
   - Copy input data to OpenVX image
   - Execute graph processing
   - Map output image to retrieve edge detection results
9. Display results using OpenCV visualization.
10. Release all OpenVX resources (graph, images, threshold, context).

## Key APIs and Concepts

- **OpenVX Context Management**: The OpenVX context is the top-level object that manages all OpenVX resources.
  - `vxCreateContext()`: Creates a new OpenVX context.
  - `vxRegisterLogCallback()`: Registers a callback function for logging OpenVX messages.
  - `vxReleaseContext()`: Releases the context and all associated resources.

- **Graph Creation and Execution**:
  - `vxCreateGraph()`: Creates a new graph within a context. Graphs define the processing pipeline.
  - `vxVerifyGraph()`: Verifies that the graph is valid and ready for execution.
  - `vxProcessGraph()`: Executes the graph to perform the defined image processing operations.
  - `vxReleaseGraph()`: Releases the graph object.

- **Image Objects**:
  - `vxCreateImage()`: Creates an image object with specified dimensions and format.
  - `vxCreateVirtualImage()`: Creates a virtual image that exists only within the graph (optimized for intermediate results).
  - `vxCopyImagePatch()`: Copies image data between host memory and OpenVX image objects.
  - `vxMapImagePatch()`: Maps an image region to host-accessible memory for reading or writing.
  - `vxUnmapImagePatch()`: Unmaps a previously mapped image region.
  - `vxReleaseImage()`: Releases an image object.

- **Threshold Configuration**:
  - `vxCreateThresholdForImage()`: Creates a threshold object for use with image processing operations.
  - `vxSetThresholdAttribute()`: Sets threshold parameters such as lower and upper bounds for hysteresis thresholding.
  - `vxReleaseThreshold()`: Releases a threshold object.

- **Processing Nodes**:
  - `vxColorConvertNode()`: Creates a node that converts between color spaces (e.g., RGB to YUV).
  - `vxChannelExtractNode()`: Creates a node that extracts a single channel from a multi-channel image.
  - `vxCannyEdgeDetectorNode()`: Creates a node that performs Canny edge detection with configurable gradient size, norm type, and threshold.
  - `vxReleaseNode()`: Releases a node object after adding it to the graph.

- **Key Enumerations and Constants**:
  - `VX_DF_IMAGE_RGB`: RGB color format (3 channels).
  - `VX_DF_IMAGE_IYUV`: YUV 4:2:0 planar format.
  - `VX_DF_IMAGE_U8`: 8-bit unsigned grayscale format.
  - `VX_CHANNEL_Y`: Luma (Y) channel identifier.
  - `VX_THRESHOLD_TYPE_RANGE`: Range threshold type for hysteresis.
  - `VX_NORM_L1`: L1 norm for gradient calculation.
  - `VX_SUCCESS`: Status code indicating successful operation.

- **Utility Functions** (from `mivisionx_utils.hpp`):
  - `ERROR_CHECK_STATUS()`: Macro to check OpenVX status codes and exit on error.
  - `ERROR_CHECK_OBJECT()`: Macro to check OpenVX object validity and exit on error.
  - `init_vx_rectangle()`: Helper to initialize rectangle structures for image regions.
  - `init_vx_image_layout_rgb()`: Helper to initialize image layout for RGB images.

## Demonstrated API Calls

### OpenVX Core

- `vxCreateContext`
- `vxReleaseContext`
- `vxRegisterLogCallback`
- `vxCreateGraph`
- `vxVerifyGraph`
- `vxProcessGraph`
- `vxReleaseGraph`
- `vxCreateImage`
- `vxCreateVirtualImage`
- `vxReleaseImage`
- `vxCopyImagePatch`
- `vxMapImagePatch`
- `vxUnmapImagePatch`
- `vxCreateThresholdForImage`
- `vxSetThresholdAttribute`
- `vxReleaseThreshold`
- `vxColorConvertNode`
- `vxChannelExtractNode`
- `vxCannyEdgeDetectorNode`
- `vxReleaseNode`
- `vxGetStatus`

### OpenVX Compatibility Extension

- `vx_ext_opencv.h` (for OpenCV interoperability)

### Data Types and Enums

- `vx_context`
- `vx_graph`
- `vx_image`
- `vx_node`
- `vx_threshold`
- `vx_status`
- `vx_rectangle_t`
- `vx_imagepatch_addressing_t`
- `vx_map_id`
- `vx_int32`
- `vx_uint8`
- `vx_uint32`
- `vx_float32`
- `vx_size`
- `VX_DF_IMAGE_RGB`
- `VX_DF_IMAGE_IYUV`
- `VX_DF_IMAGE_U8`
- `VX_CHANNEL_Y`
- `VX_THRESHOLD_TYPE_RANGE`
- `VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER`
- `VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER`
- `VX_NORM_L1`
- `VX_SUCCESS`
- `VX_WRITE_ONLY`
- `VX_READ_ONLY`
- `VX_MEMORY_TYPE_HOST`
- `VX_NOGAP_X`

### OpenCV (for visualization)

- `cv::Mat`
- `cv::VideoCapture`
- `cv::resize`
- `cv::imshow`
- `cv::waitKey`
- `cv::imread`
