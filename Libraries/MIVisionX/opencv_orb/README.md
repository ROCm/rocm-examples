# MIVisionX ORB Feature Detection

> [!WARNING]
> This example is deprecated. OpenCV support from MIVisionX will be removed in future releases, this example will only work on Ubuntu 22.04 with MIVisionX built from source with OpenCV support. For more details on how to build MIVisionX, please see https://github.com/ROCm/MIVisionX and https://github.com/ROCm/MIVisionX/pull/1575

## Description

This example demonstrates ORB (Oriented FAST and Rotated BRIEF) feature detection using OpenVX with OpenCV extensions. The example processes images or live camera feeds to detect and extract keypoint features, which are useful for object recognition, image matching, and tracking applications.

ORB is a fast and efficient feature detector that combines:

- FAST (Features from Accelerated Segment Test) keypoint detection
- BRIEF (Binary Robust Independent Elementary Features) descriptor computation
- Orientation compensation for rotation invariance

## Application flow

1. Parse command-line arguments for input source and image dimensions.
2. Create OpenVX context and register log callback.
3. Load OpenCV extension kernels into the OpenVX context.
4. Create OpenVX graph for the feature detection pipeline.
5. Allocate grayscale image for input and keypoints array for output.
6. Configure ORB detector parameters (number of features, scale factor, pyramid levels, etc.).
7. Create ORB detection node with specified parameters.
8. Verify the graph to ensure all connections are valid.
9. Process input (either from image file or live camera feed):
   - Convert input to grayscale if needed
   - Resize to target dimensions
   - Copy input data to OpenVX image
   - Execute graph to detect features
   - Query the number of detected keypoints
   - Map keypoints array to retrieve feature locations
10. Visualize detected keypoints by drawing circles on the output image.
11. Release all OpenVX resources (graph, array, image, context).

## Key APIs and Concepts

- **OpenVX Context Management**: The OpenVX context manages all OpenVX resources and loaded extensions.
  - `vxCreateContext()`: Creates a new OpenVX context.
  - `vxRegisterLogCallback()`: Registers a callback function for logging OpenVX messages.
  - `vxLoadKernels()`: Loads extension kernel libraries (e.g., `vx_opencv` for OpenCV integration).
  - `vxReleaseContext()`: Releases the context and all associated resources.

- **Graph Creation and Execution**:
  - `vxCreateGraph()`: Creates a new graph within a context for defining the processing pipeline.
  - `vxVerifyGraph()`: Verifies that the graph is valid and ready for execution.
  - `vxProcessGraph()`: Executes the graph to perform feature detection.
  - `vxReleaseGraph()`: Releases the graph object.

- **Image Objects**:
  - `vxCreateImage()`: Creates an image object with specified dimensions and format (grayscale for ORB input).
  - `vxCopyImagePatch()`: Copies image data between host memory and OpenVX image objects.
  - `vxReleaseImage()`: Releases an image object.

- **Array Objects for Keypoints**:
  - `vxCreateArray()`: Creates an array to store detected keypoints. Specifies element type (`VX_TYPE_KEYPOINT`) and capacity.
  - `vxQueryArray()`: Queries array properties such as the number of items (`VX_ARRAY_NUMITEMS`).
  - `vxMapArrayRange()`: Maps a range of array elements to host-accessible memory for reading keypoint data.
  - `vxUnmapArrayRange()`: Unmaps a previously mapped array range.
  - `vxReleaseArray()`: Releases an array object.

- **OpenCV Extension - ORB Detection**:
  - `vxExtCvNode_orbDetect()`: Creates an ORB feature detection node with configurable parameters:
    - `n_features`: Maximum number of features to detect
    - `scale_factor`: Pyramid decimation ratio (typically 1.2)
    - `n_levels`: Number of pyramid levels
    - `edge_threshold`: Size of border where features are not detected
    - `first_level`: Level of pyramid to start feature detection
    - `wta_k`: Number of points for computing BRIEF descriptor (2, 3, or 4)
    - `score_type`: Scoring method (0 for HARRIS, 1 for FAST)
    - `patch_size`: Size of patch used for descriptor computation

- **Keypoint Data Structure**:
  - `vx_keypoint_t`: Structure containing keypoint information including x, y coordinates, strength, scale, orientation, and tracking status.

- **Key Enumerations and Constants**:
  - `VX_DF_IMAGE_U8`: 8-bit unsigned grayscale image format.
  - `VX_TYPE_KEYPOINT`: Data type identifier for keypoint structures.
  - `VX_ARRAY_NUMITEMS`: Array attribute for querying the number of valid items.
  - `VX_SUCCESS`: Status code indicating successful operation.
  - `VX_READ_ONLY`: Access mode for read-only mapping.
  - `VX_MEMORY_TYPE_HOST`: Memory type for host-accessible data.

- **Utility Functions** (from `mivisionx_utils.hpp`):
  - `ERROR_CHECK_STATUS()`: Macro to check OpenVX status codes and exit on error.
  - `ERROR_CHECK_OBJECT()`: Macro to check OpenVX object validity and exit on error.
  - `init_vx_rectangle()`: Helper to initialize rectangle structures for image regions.
  - `init_vx_image_layout_gray()`: Helper to initialize image layout for grayscale images.

## Demonstrated API Calls

### OpenVX Core

- `vxCreateContext`
- `vxReleaseContext`
- `vxRegisterLogCallback`
- `vxLoadKernels`
- `vxCreateGraph`
- `vxVerifyGraph`
- `vxProcessGraph`
- `vxReleaseGraph`
- `vxCreateImage`
- `vxReleaseImage`
- `vxCopyImagePatch`
- `vxCreateArray`
- `vxQueryArray`
- `vxMapArrayRange`
- `vxUnmapArrayRange`
- `vxReleaseArray`
- `vxReleaseNode`
- `vxGetStatus`

### OpenCV Extension

- `vxExtCvNode_orbDetect`
- `vx_opencv` kernel library

### Data Types and Enums

- `vx_context`
- `vx_graph`
- `vx_image`
- `vx_node`
- `vx_array`
- `vx_keypoint_t`
- `vx_status`
- `vx_rectangle_t`
- `vx_imagepatch_addressing_t`
- `vx_map_id`
- `vx_int32`
- `vx_uint8`
- `vx_uint32`
- `vx_float32`
- `vx_size`
- `VX_DF_IMAGE_U8`
- `VX_TYPE_KEYPOINT`
- `VX_ARRAY_NUMITEMS`
- `VX_SUCCESS`
- `VX_WRITE_ONLY`
- `VX_READ_ONLY`
- `VX_MEMORY_TYPE_HOST`

### OpenCV (for visualization and preprocessing)

- `cv::Mat`
- `cv::VideoCapture`
- `cv::imread`
- `cv::cvtColor`
- `cv::resize`
- `cv::circle`
- `cv::imshow`
- `cv::waitKey`
- `cv::Point`
- `cv::Size`
- `cv::Scalar`
- `COLOR_RGB2GRAY`
