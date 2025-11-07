# rocAL Video and Sequence Processing

## Description

This example demonstrates rocAL's video processing capabilities for handling video files and image sequences with various reader types, hardware acceleration options, and sequence manipulation features.

The operation performs video/sequence processing:

$Output_{sequence} = \text{Process}(\text{Decode}(\text{Load}(Video/Sequence)))$

where

- $Video/Sequence$ is the input video file or image sequence
- $\text{Load}$ reads video frames or sequence images based on reader type
- $\text{Decode}$ performs hardware or software decoding
- $\text{Process}$ applies optional augmentations and sequence operations
- $Output_{sequence}$ contains processed frame sequences

## Application flow

1. Parse command-line arguments for video/sequence parameters, reader type, and processing options.
2. Create rocAL context with specified processing mode and batch configuration.
3. Configure video/sequence processing based on reader case:
   - **Video Reader**: Load video files with hardware/software decoding
   - **Video Resize Reader**: Load and resize videos during decoding
   - **Sequence Reader**: Load image sequences from directories
   - **Single Shard Variants**: Process specific data shards
4. Set up decoder configuration:
   - Choose hardware decoding (ROCDECODE) or software decoding (FFMPEG)
   - Configure sequence parameters (length, step, stride)
   - Enable metadata, frame numbers, and timestamp options
5. Apply optional sequence processing:
   - Sequence rearrangement for temporal augmentation
   - Dynamic parameter adjustments during processing
6. Verify the augmentation graph for correct configuration.
7. Process video/sequence data:
   - Run processing pipeline for each batch
   - Extract frame sequences with specified parameters
   - Copy processed data to host memory
8. Handle metadata and temporal information:
   - Retrieve frame numbers and timestamps if enabled
   - Process video labels if metadata reading is enabled
9. Save processed frames or create output videos if requested.
10. Collect and display detailed timing information.
11. Clean up rocAL context and release resources.

## Key APIs and Concepts

- **Multiple Reader Types**: rocAL provides various readers for different video and sequence formats, each optimized for specific use cases.

- **Video File Reading**:
  - `rocalVideoFileSource()`: Loads video files with configurable decoder settings and sequence parameters.
  - `rocalVideoFileSourceSingleShard()`: Processes video files in single shard mode for parallel processing.
  - `rocalVideoFileResize()`: Loads and resizes videos during decoding for memory efficiency.
  - `rocalVideoFileResizeSingleShard()`: Single shard variant with resize capability.

- **Image Sequence Processing**:
  - `rocalSequenceReader()`: Loads image sequences from directory structures.
  - `rocalSequenceReaderSingleShard()`: Single shard sequence processing for parallel operations.

- **Hardware Acceleration**:
  - `RocalDecodeDevice`: Enumeration for hardware vs software decoding (`ROCAL_HW_DECODE`, `ROCAL_SW_DECODE`).
  - `RocalDecoderType`: Specifies decoder implementations (`ROCAL_DECODER_VIDEO_ROCDECODE`, `ROCAL_DECODER_VIDEO_FFMPEG_SW`).

- **Video Metadata Handling**:
  - `rocalCreateVideoLabelReader()`: Creates label readers for video metadata processing.
  - `rocalGetImageLabels()`: Retrieves label information for video sequences.

- **Sequence Manipulation**:
  - `rocalSequenceRearrange()`: Rearranges frame sequences for temporal augmentation.
  - Supports custom frame ordering patterns for varied temporal representations.

- **Temporal Information**:
  - `rocalGetSequenceStartFrameNumber()`: Retrieves start frame numbers for sequences.
  - `rocalGetSequenceFrameTimestamps()`: Gets timestamp information for video frames.

- **Sequence Configuration**:
  - Sequence length controls number of frames per processed sequence.
  - Frame step determines sampling interval between frames.
  - Frame stride controls temporal resolution of extracted sequences.

- **Multi-format Support**:
  - Supports various video formats through FFMPEG software decoding.
  - Hardware acceleration available through ROCDECODE for supported formats.
  - Image sequence support for frame-by-frame processing workflows.

- **Output Options**:
  - Individual frame saving with sequence-aware naming.
  - Video file generation per sequence for complete output preservation.

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
- `rocalResetLoaders`

### Video File Reading

- `rocalVideoFileSource`
- `rocalVideoFileSourceSingleShard`
- `rocalVideoFileResize`
- `rocalVideoFileResizeSingleShard`

### Image Sequence Processing

- `rocalSequenceReader`
- `rocalSequenceReaderSingleShard`

### Video Metadata Handling

- `rocalCreateVideoLabelReader`
- `rocalGetImageLabels`

### Sequence Manipulation

- `rocalSequenceRearrange`

### Temporal Information

- `rocalGetSequenceStartFrameNumber`
- `rocalGetSequenceFrameTimestamps`

### Information and Metadata

- `rocalGetOutputWidth`
- `rocalGetOutputHeight`
- `rocalGetOutputColorFormat`
- `rocalGetAugmentationBranchCount`
- `rocalIsEmpty`
- `rocalGetRemainingImages`
- `rocalGetImageNameLen`
- `rocalGetImageName`

### Dynamic Parameter Management

- `rocalCreateIntParameter`
- `rocalUpdateIntParameter`
- `rocalGetIntValue`

### Data Types and Enums

- `RocalContext`
- `RocalTensor`
- `RocalProcessMode`
- `RocalImageColor`
- `RocalDecoderType`
- `RocalDecodeDevice`
- `RocalIntParam`
- `RocalTimingInfo`
- `ROCAL_COLOR_RGB24`
- `ROCAL_COLOR_U8`
- `ROCAL_DECODER_VIDEO_FFMPEG_SW`
- `ROCAL_DECODER_VIDEO_ROCDECODE`
- `ROCAL_DECODER_OPENCV`
- `ROCAL_HW_DECODE`
- `ROCAL_SW_DECODE`
- `ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED`
- `ROCAL_PROCESS_GPU`
- `ROCAL_PROCESS_CPU`
- `ROCAL_OK`
