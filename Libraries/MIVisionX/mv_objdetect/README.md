# MIVisionX Object Detection

## Description

This example demonstrates real-time object detection using the MIVisionX deployment framework with a pre-trained YoloV2 Tiny model. The example showcases automated model compilation, video decoding integration, and efficient inference execution on AMD GPUs.

The example uses **YoloV2 Tiny trained on 20 PASCAL VOC classes**: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, and tvmonitor.

This implementation **automatically downloads and compiles the model** during the build process using the `mv_compile` tool, providing a streamlined development experience.

## Application flow

1. Parse command-line arguments for input source, detection parameters, and visualization options.
2. Initialize MIVisionX deployment system with installation folder path.
3. Query inference model configuration to obtain input/output dimensions and tensor information.
4. Parse model configuration to extract tensor dimensions for all inputs and outputs.
5. If processing video input, register preprocessing callback to integrate video decoder node into the graph.
6. Create inference session with specified memory type (host or device).
7. Allocate host memory for input and output tensors based on queried dimensions.
8. For image input, load and preprocess image data (resize, color conversion, normalization).
9. Transfer input data to inference session using `mvSetInputDataFromMemory`.
10. Initialize postprocessing module for bounding box detection with NMS parameters.
11. Execute inference loop:
    - Run inference with `mvRunInference`
    - Retrieve output data with `mvGetOutputData`
    - Extract bounding box detections using postprocessing APIs
    - Optionally compute argmax for classification results
    - Visualize detections if enabled
12. Report performance metrics (average inference time).
13. Shutdown postprocessing module and release inference session.
14. Clean up allocated memory and shutdown deployment system.

## Key APIs and Concepts

- **Deployment System Initialization**: The MIVisionX deployment framework provides a high-level API for running compiled neural network models.
  - `mvInitializeDeployment()`: Initializes the deployment system with the installation folder containing compiled model artifacts.
  - `mvShutdown()`: Shuts down the deployment system and releases all resources.

- **Model Configuration Query**:
  - `QueryInference()`: Queries the compiled model to retrieve the number of inputs, outputs, and their configuration string containing dimensions and names.

- **Session Management**:
  - `mvCreateInferenceSession()`: Creates an inference session for executing the model. Accepts installation folder path and memory type (host or device).
  - `mvReleaseInferenceSession()`: Releases the inference session and associated resources.

- **Data Transfer**:
  - `mvSetInputDataFromMemory()`: Transfers input tensor data from host memory to the inference session. Requires input index, data pointer, size in bytes, and memory type.
  - `mvGetOutputData()`: Retrieves output tensor data from the inference session to host memory after inference execution.

- **Inference Execution**:
  - `mvRunInference()`: Executes the inference graph. Returns execution time in milliseconds and supports multiple iterations for performance testing.

- **Preprocessing Integration**:
  - `SetPreProcessCallback()`: Registers a callback function to add custom preprocessing nodes (e.g., video decoder) to the inference graph.
  - Preprocessing callback receives the inference session, output tensor, and custom arguments to build the preprocessing pipeline.

- **Postprocessing for Object Detection**:
  - `mv_postproc_init()`: Initializes the postprocessing module for object detection with parameters including number of classes, grid size, anchor biases, confidence threshold, NMS threshold, and input dimensions.
  - `mv_postproc_getBB_detections()`: Extracts bounding box detections from the model output tensor. Applies confidence filtering and non-maximum suppression (NMS).
  - `mv_postproc_argmax()`: Computes top-K classification results using argmax operation on output probabilities.
  - `mv_postproc_shutdown()`: Shuts down the postprocessing module and releases resources.

- **OpenVX Integration for Video Decoding**: The preprocessing callback uses OpenVX to integrate video decoding.
  - `vxCreateContext()`: Creates OpenVX context from the inference handle.
  - `vxLoadKernels()`: Loads AMD media extension kernels for video decoding.
  - `amdMediaDecoderNode()`: Creates a video decoder node that outputs decoded frames as OpenVX images.
  - `vxConvertImageToTensorNode()`: Converts OpenVX image to tensor format with normalization (scale and bias factors).

- **Key Data Structures**:
  - `mivid_session`: Opaque handle to an inference session.
  - `mivid_handle`: Internal handle structure containing OpenVX context and graph.
  - `mv_preprocess_callback_args`: Structure for passing arguments to preprocessing callback (decoder string, loop flag, normalization factors).
  - `BBox`: Bounding box structure containing position (x, y), dimensions (w, h), confidence, label, and image number.
  - `ClassLabel`: Classification result structure with class index and probability.

- **Memory Types**:
  - `mv_mem_type_host`: Data resides in host (CPU) memory.
  - `mv_mem_type_device`: Data resides in device (GPU) memory.

## Demonstrated API Calls

### MIVisionX Deployment

- `mvInitializeDeployment`
- `mvShutdown`
- `QueryInference`
- `mvCreateInferenceSession`
- `mvReleaseInferenceSession`
- `mvSetInputDataFromMemory`
- `mvGetOutputData`
- `mvRunInference`
- `SetPreProcessCallback`

### MIVisionX Postprocessing

- `mv_postproc_init`
- `mv_postproc_getBB_detections`
- `mv_postproc_argmax`
- `mv_postproc_shutdown`

### OpenVX Core (used in preprocessing callback)

- `vxCreateContext`
- `vxCreateGraph`
- `vxCreateImage`
- `vxCreateVirtualImage`
- `vxLoadKernels`
- `vxQueryTensor`
- `vxConvertImageToTensorNode`
- `vxMapImagePatch`
- `vxUnmapImagePatch`
- `vxQueryImage`
- `vxGetStatus`

### AMD Media Extension

- `amdMediaDecoderNode`
- `vx_amd_media` kernel library

### Data Types and Enums

- `mivid_session`
- `mivid_handle`
- `mv_status`
- `mv_mem_type_host`
- `mv_mem_type_device`
- `mv_preprocess_callback_args`
- `BBox`
- `ClassLabel`
- `vx_context`
- `vx_graph`
- `vx_image`
- `vx_node`
- `vx_tensor`
- `vx_status`
- `vx_size`
- `vx_uint32`
- `vx_uint8`
- `vx_float32`
- `vx_imagepatch_addressing_t`
- `vx_rectangle_t`
- `vx_map_id`
- `VX_SUCCESS`
- `VX_ERROR_GRAPH_ABANDONED`
- `VX_TENSOR_NUMBER_OF_DIMS`
- `VX_TENSOR_DIMS`
- `VX_READ_ONLY`
- `VX_WRITE_ONLY`
- `VX_MEMORY_TYPE_HOST`
- `VX_NOGAP_X`
- `MV_SUCCESS`
- `MV_ERROR_GRAPH_ABANDONED`

### OpenCV (for visualization)

- `cv::Mat`
- `cv::imread`
- `cv::resize`
- `cv::imshow`
- `cv::waitKey`
- `cv::rectangle`
- `cv::putText`
- `cv::getTextSize`
- `cv::Point`
- `cv::Size`
- `cv::Scalar`
