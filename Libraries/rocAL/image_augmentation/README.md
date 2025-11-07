# rocAL Image Augmentation Pipeline

## Description

This example demonstrates advanced rocAL augmentation capabilities with complex multi-branch processing pipelines, dynamic parameter adjustment, and various image transformation effects.

The operation performs multiple parallel augmentation streams:

$Output_{branch_1} = \\text{Rotate}(\\text{FishEye}(\\text{Rain}(\\text{Resize}(Input))))$

$Output_{branch_2} = \\underbrace{\\text{Blur}(\\text{Blur}(\\dots \\text{Blur}(\\text{CropResize}(Input))))}_{\\text{aug\\_depth\\ times}}$

$Output_{branch_3} = \\text{Exposure}(\\text{Blend}(Input, \\text{Snow}(Input)))$

where

- $Input$ is the original image loaded from disk
- Each branch represents a different augmentation strategy
- Multiple output variations are generated per input image for data augmentation

## Application flow

1. Parse command-line arguments for input path, augmentation parameters, batch size, and processing mode.
2. Create rocAL context with specified configuration (GPU or CPU processing).
3. Set up augmentation parameters:
   - Create uniform random parameters for crop area adjustment
   - Create discrete value parameters for rotation angles
   - Initialize dynamic color temperature adjustment parameter
4. Configure augmentation pipeline:
   - Create JPEG file source with specified dimensions and decoder type
   - Apply initial resize operation to standardize input dimensions
   - Set up multiple augmentation branches:
     - Branch 1: Rain → Fisheye → Rotation
     - Branch 2: CropResize → Multiple Blur passes
     - Branch 3: Snow → Blend → Exposure
5. Verify the augmentation graph for correct configuration.
6. Process images with dynamic augmentation:
   - Run the augmentation pipeline for each batch
   - Update color temperature parameter dynamically during processing
   - Retrieve output tensors from all augmentation branches
   - Copy processed data to host memory
7. Save augmented images with branch-specific naming conventions if enabled.
8. Collect and display detailed timing information.
9. Clean up rocAL context and release resources.

## Key APIs and Concepts

- **Multi-branch Augmentation**: rocAL supports multiple parallel augmentation branches, generating diverse output variations from a single input image.

- **Dynamic Parameter Management**:
  - `rocalCreateFloatUniformRand()`: Creates uniformly distributed random float parameters for configurable augmentation intensity.
  - `rocalUpdateFloatUniformRand()`: Dynamically updates random parameter ranges during processing.
  - `rocalCreateFloatRand()`: Creates discrete value parameters with custom frequency distributions.
  - `rocalCreateIntParameter()`: Creates integer parameters for dynamic adjustment.
  - `rocalUpdateIntParameter()`: Updates integer parameters during runtime for dynamic behavior.

- **Advanced Augmentation Operations**:
  - `rocalRain()`: Simulates rain effect on images for weather augmentation.
  - `rocalFishEye()`: Applies fisheye lens distortion for geometric augmentation.
  - `rocalRotate()`: Rotates images with random angles for orientation variation.
  - `rocalCropResize()`: Performs cropping with random area parameters followed by resizing.
  - `rocalBlur()`: Applies blur effect with configurable depth for image smoothing.
  - `rocalSnow()`: Simulates snow effect for weather-based augmentation.
  - `rocalBlend()`: Blends two images with configurable mixing ratio.
  - `rocalExposure()`: Adjusts image exposure with dynamic parameters.

- **Tensor Management**:
  - `rocalGetOutputTensors()`: Retrieves all output tensors from augmentation branches.
  - Each tensor contains processed data from different augmentation paths.
  - Tensors are copied to host memory sequentially to form final output.

- **Performance Analysis**:
  - `rocalGetTimingInfo()`: Provides comprehensive timing breakdown including load, decode, process, and transfer phases.
  - Detailed performance metrics help identify bottlenecks in complex augmentation pipelines.

- **Branch Configuration**:
  - Multiple augmentation branches can be configured independently.
  - Each branch can have different operation sequences and parameters.
  - Output dimensions are automatically calculated based on branch count and batch size.

- **Output Organization**:
  - Output data is organized with batch dimension first, then branch dimension.
  - Individual images can be extracted using calculated offsets and dimensions.
  - Branch-specific naming conventions identify augmentation variations.

## Demonstrated API Calls

### rocAL Core Functions

- `rocalCreate`
- `rocalRelease`
- `rocalVerify`
- `rocalRun`
- `rocalGetStatus`
- `rocalGetErrorMessage`
- `rocalCopyToOutput`
- `rocalGetTimingInfo`

### Image Loading

- `rocalJpegFileSource`

### Advanced Augmentation Operations

- `rocalResize`
- `rocalRain`
- `rocalFishEye`
- `rocalRotate`
- `rocalCropResize`
- `rocalBlur`
- `rocalSnow`
- `rocalBlend`
- `rocalExposure`

### Dynamic Parameter Management

- `rocalCreateFloatUniformRand`
- `rocalUpdateFloatUniformRand`
- `rocalCreateFloatRand`
- `rocalCreateIntParameter`
- `rocalUpdateIntParameter`
- `rocalGetIntValue`

### Tensor and Output Management

- `rocalGetOutputTensors`
- `rocalGetOutputWidth`
- `rocalGetOutputHeight`
- `rocalGetOutputColorFormat`
- `rocalGetAugmentationBranchCount`

### Information and Metadata

- `rocalIsEmpty`
- `rocalGetRemainingImages`

### Data Types and Enums

- `RocalContext`
- `RocalTensor`
- `RocalTensorList`
- `RocalProcessMode`
- `RocalImageColor`
- `RocalDecoderType`
- `RocalSizePolicy`
- `RocalFloatParam`
- `RocalIntParam`
- `RocalTimingInfo`
- `ROCAL_COLOR_RGB24`
- `ROCAL_COLOR_U8`
- `ROCAL_DECODER_OPENCV`
- `ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED`
- `ROCAL_PROCESS_GPU`
- `ROCAL_PROCESS_CPU`
- `ROCAL_OK`
