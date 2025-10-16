# rocDecode Video to Frame Sequence

## Description

This example demonstrates extracting a complete frame sequence from a video file using the rocDecode library. The sample decodes all frames from a video and saves them as a continuous sequence, useful for creating training datasets, video analysis, or frame-by-frame processing workflows. It provides options for output format selection and frame numbering.

## Application Flow

1. Parse command-line arguments for input file, output directory, device ID, and sequence options.
2. Initialize the FFMPEG video demuxer to extract codec information.
3. Create the video decoder instance with the specified codec and device.
4. Verify codec support on the selected GPU device.
5. Create output directory structure for frame sequence.
6. Loop through entire video stream:
   - Demux video packets from input file.
   - Decode frames using hardware acceleration.
   - Retrieve each decoded frame in sequence.
   - Save frame with sequential numbering (e.g., 0000.yuv, 0001.yuv, 0002.yuv).
   - Maintain frame order for sequence integrity.
7. Display total number of frames in sequence.
8. Clean up decoder and demuxer resources.

## Key APIs and Concepts

- **Sequential Frame Extraction**: Processes and saves all frames in order:
  - Maintains decode order for frame sequence.
  - Sequential file naming for easy processing.
  - Complete video coverage without frame skipping.
  - Preserves temporal relationships between frames.

- **Frame Sequence Output**: Organized frame storage:
  - Zero-padded frame numbers for proper sorting.
  - Consistent naming convention across sequence.
  - Raw YUV format preserves decode quality.
  - Directory structure for organized storage.

- **Decoder Configuration**: Standard setup optimized for sequential processing:
  - `rocDecCreateDecoder()`: Initializes decoder with codec parameters.
  - Output surface memory configured for efficient frame retrieval.
  - Display delay set to maintain frame order.

- **Frame Management**: Efficient processing of frame sequence:
  - `rocDecGetVideoFrame()`: Retrieves frames in decode order.
  - Frames are saved immediately after retrieval.
  - Memory is released promptly to maintain decoder surface pool.
  - Continuous processing without frame buffering.

- **Use Cases**:
  - Creating training datasets for machine learning.
  - Video analysis requiring frame-by-frame access.
  - Extracting complete frame sequences for processing pipelines.
  - Quality analysis and comparison workflows.
  - Preparing data for computer vision applications.
  - Video editing and post-production workflows.

- **Output Organization**:
  - Frames saved in specified output directory.
  - Sequential numbering starting from 0000.
  - Raw YUV format (NV12, P016, YUV444, etc.).
  - Metadata preserved for reconstruction.

## Demonstrated API Calls

### rocDecode APIs

- `rocDecCreateDecoder`
- `rocDecDecodeFrame`
- `rocDecGetVideoFrame`
- `rocDecGetDecodeStatus`
- `rocDecDestroyDecoder`
- `rocDecCreateVideoParser`
- `rocDecParseVideoData`
- `rocDecDestroyVideoParser`
- `rocDecGetErrorName`

### HIP Runtime APIs

- `hipSetDevice`
- `hipGetDeviceProperties`
- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyDtoH`

### FFMPEG APIs

- `avformat_open_input`
- `avformat_find_stream_info`
- `av_find_best_stream`
- `av_read_frame`
- `av_packet_alloc`
- `av_packet_free`
- `av_packet_unref`
- `avformat_close_input`
- `av_bsf_get_by_name`
- `av_bsf_alloc`
- `av_bsf_init`
- `av_bsf_send_packet`
- `av_bsf_receive_packet`
- `av_bsf_free`

### Data Types and Enums

- `rocDecDecoderHandle`
- `RocdecVideoParser`
- `rocDecVideoCodec`
- `rocDecVideoSurfaceFormat`
- `rocDecVideoChromaFormat`
- `rocDecDecoderCreateInfo`
- `RocdecParserParams`
- `RocdecVideoFormat`
- `RocdecPicParams`
- `RocdecParserDispInfo`
- `rocDecDecodeStatus`
- `AVCodecID`
- `AVFormatContext`
- `AVPacket`
