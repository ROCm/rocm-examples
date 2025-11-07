# rocAL Basic Image Processing

## Description

This example demonstrates basic rocAL functionality for loading and processing JPEG images with simple crop and resize augmentations.

The operation performs the following transformation:

$Output = \text{Resize}(\text{Crop}(Input))$

where

- $Input$ is a JPEG image loaded from disk
- $\text{Crop}$ extracts a region from the input image with adjustable aspect ratio
- $\text{Resize}$ rescales the cropped region to a fixed size (224×224 by default)

## Application flow

1. Parse command-line arguments for input path, batch size, output dimensions, and processing mode.
2. Create rocAL context with specified batch size and processing mode (GPU or CPU).
3. Set up the augmentation pipeline:
   - Create JPEG file source from specified input directory
   - Configure color format (RGB or grayscale)
   - Apply crop and resize augmentation with specified parameters
4. Verify the augmentation graph for correct configuration.
5. Process images in batches:
   - Run the augmentation pipeline for each batch
   - Copy processed images from device to host memory
   - Save individual images to disk with batch and iteration information
6. Reset loaders and continue processing for multiple batches if in dynamic mode.
7. Clean up rocAL context and release resources.

## Key APIs and Concepts

- **rocAL Context Creation**: The rocAL processing environment is initialized using `rocalCreate()` with parameters for batch size, processing mode, GPU ID, and number of streams.

- **Image Loading**:
  - `rocalJpegFileSource()`: Creates a JPEG image source that loads images from a specified directory. This function can automatically determine optimal decode sizes or use user-specified dimensions.
  - `RocalImageColor`: Enumeration defining color formats (e.g., `ROCAL_COLOR_RGB24` for 24-bit RGB, `ROCAL_COLOR_U8` for 8-bit grayscale).
  - `RocalDecoderType`: Specifies the decoder type, with `ROCAL_DECODER_OPENCV` being the default for this example.

- **Augmentation Operations**:
  - `rocalCropResizeFixed()`: Performs cropping with adjustable aspect ratio followed by resizing to fixed dimensions. The function includes randomization parameters for robust augmentation.
  - `RocalSizePolicy`: Policy for size selection (`ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED` for user-specified dimensions).

- **Processing and Execution**:
  - `rocalRun()`: Executes the augmentation pipeline for the current batch.
  - `rocalCopyToOutput()`: Copies processed image data from device to host memory.
  - `rocalVerify()`: Validates the augmentation graph configuration before processing.

- **Batch Processing**:
  - `rocalGetOutputWidth()` / `rocalGetOutputHeight()`: Retrieve output image dimensions.
  - `rocalGetOutputColorFormat()`: Get the color format of processed images.
  - `rocalGetAugmentationBranchCount()`: Get the number of augmentation branches (output variations).

- **Dynamic Processing**:
  - `rocalIsEmpty()`: Check if there are remaining images to process.
  - `rocalGetRemainingImages()`: Get count of remaining images in the dataset.
  - `rocalResetLoaders()`: Reset image loaders for multiple processing runs.

- **Image Metadata**:
  - `rocalGetImageNameLen()`: Get the length of image name strings for the current batch.
  - `rocalGetImageName()`: Retrieve the original image filenames for processed images.

- **Error Handling**:
  - `rocalGetStatus()`: Get the current status of rocAL operations.
  - `rocalGetErrorMessage()`: Retrieve detailed error messages for debugging.

- **Resource Management**:
  - `rocalRelease()`: Clean up and release rocAL resources and memory.

## Demonstrated API Calls

### rocAL Core Functions

- `rocalCreate`
- `rocalRelease`
- `rocalVerify`
- `rocalRun`
- `rocalGetStatus`
- `rocalGetErrorMessage`
- `rocalCopyToOutput`
- `rocalResetLoaders`

### Image Loading

- `rocalJpegFileSource`

### Augmentation Operations

- `rocalCropResizeFixed`

### Information and Metadata

- `rocalGetOutputWidth`
- `rocalGetOutputHeight`
- `rocalGetOutputColorFormat`
- `rocalGetAugmentationBranchCount`
- `rocalIsEmpty`
- `rocalGetRemainingImages`
- `rocalGetImageNameLen`
- `rocalGetImageName`

### Data Types and Enums

- `RocalContext`
- `RocalTensor`
- `RocalProcessMode`
- `RocalImageColor`
- `RocalDecoderType`
- `RocalSizePolicy`
- `ROCAL_COLOR_RGB24`
- `ROCAL_COLOR_U8`
- `ROCAL_DECODER_OPENCV`
- `ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED`
- `ROCAL_PROCESS_GPU`
- `ROCAL_PROCESS_CPU`
- `ROCAL_OK`
