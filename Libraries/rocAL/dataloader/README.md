# rocAL Multi-threaded Data Loading

## Description

This example demonstrates rocAL's multi-threaded data loading capabilities for efficient parallel processing of large datasets. The example shows how to distribute data loading across multiple threads (shards) and handle metadata with labels.

The operation performs the following processing:

$Output_{shard_i} = \\text{Resize}(\\text{Load}_{shard_i}(Input))$

where

- $Input$ is the dataset directory containing images and labels
- $\\text{Load}_{shard_i}$ loads a subset of data in thread $i$ (where $i \\in [0, N-1]$ for $N$ shards)
- $\\text{Resize}$ rescales images to a uniform size (224×224 by default)
- $Output_{shard_i}$ contains processed data from shard $i$

## Application flow

1. Parse command-line arguments for input path, number of shards, GPU configuration, and processing parameters.
2. Create and launch multiple threads (one per shard):
   - Each thread creates its own rocAL context with specific GPU assignment
   - Configure single shard JPEG file source with shard-specific parameters
   - Create label reader for metadata handling if enabled
   - Apply resize augmentation to standardize image dimensions
3. Verify the augmentation graph in each thread.
4. Process data concurrently across all shards:
   - Each thread processes its assigned subset of images
   - Load images and corresponding labels from the dataset
   - Apply resize transformation and copy to host memory
   - Optionally save processed images with shard-specific naming
5. Collect timing information from each thread for performance analysis.
6. Join all threads and wait for completion.
7. Clean up rocAL contexts and release resources in each thread.

## Key APIs and Concepts

- **Multi-threading Setup**: rocAL supports concurrent processing through separate contexts per thread, enabling efficient parallel data loading and augmentation.

- **Single Shard Data Loading**:
  - `rocalJpegFileSourceSingleShard()`: Creates a JPEG image source for a specific shard, enabling parallel data loading across multiple threads.
  - Shard parameters include shard ID, total shard count, and optional shuffling within each shard.
  - Each shard processes a unique subset of the dataset for load balancing.

- **Label Reader Integration**:
  - `rocalCreateLabelReader()`: Creates a label reader that associates images with metadata labels from label files.
  - `rocalGetImageLabels()`: Retrieves label information for processed images, useful for supervised learning scenarios.

- **Thread Safety**:
  - Each thread maintains its own rocAL context to ensure thread-safe operations.
  - Mutex synchronization is used for context creation to prevent OpenVX kernel loading conflicts.

- **Performance Monitoring**:
  - `rocalGetTimingInfo()`: Provides detailed timing breakdown including load, decode, process, and transfer times.
  - Each thread reports its own performance metrics for comprehensive analysis.

- **Data Distribution**:
  - Shards automatically distribute data evenly across threads for balanced processing.
  - Optional shuffling can be enabled per shard for randomized data access patterns.

- **GPU Resource Management**:
  - Multiple threads can share GPUs using round-robin assignment.
  - Each context specifies GPU ID for targeted execution.

- **Batch Processing per Shard**:
  - Each shard processes batches independently with its own batch size.
  - Thread-specific output naming prevents file conflicts.

- **Standard Augmentation**:
  - `rocalResize()`: Applies uniform resizing to all images across all shards.
  - Maintains consistent output dimensions regardless of input image sizes.

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

### Multi-threaded Data Loading

- `rocalJpegFileSourceSingleShard`

### Label and Metadata Handling

- `rocalCreateLabelReader`
- `rocalGetImageLabels`

### Augmentation Operations

- `rocalResize`

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
- `RocalTimingInfo`
- `ROCAL_COLOR_RGB24`
- `ROCAL_COLOR_U8`
- `ROCAL_DECODER_OPENCV`
- `ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED`
- `ROCAL_PROCESS_GPU`
- `ROCAL_PROCESS_CPU`
- `ROCAL_OK`
