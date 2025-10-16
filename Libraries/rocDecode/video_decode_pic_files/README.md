# rocDecode Picture File Extraction

## Description

This example demonstrates extracting individual decoded frames from a video file and saving them as separate picture files. The sample decodes a video stream using the rocDecode library and writes each decoded frame to disk as an individual YUV file, making it useful for frame-by-frame analysis, quality inspection, or creating image sequences from video content.

## Application Flow

1. Parse command-line arguments for input file, output directory, device ID, and frame options.
2. Initialize the FFMPEG video demuxer to extract codec information.
3. Create the video decoder instance with the specified codec and device.
4. Verify codec support on the selected GPU device.
5. Create output directory if it doesn't exist.
6. Loop through video stream:
   - Demux video packets from input file.
   - Decode frames using hardware acceleration.
   - Retrieve each decoded frame.
   - Save frame to individual file with sequential naming (e.g., frame_0000.yuv, frame_0001.yuv).
   - Release frame back to decoder.
7. Display total number of frames extracted.
8. Clean up decoder and demuxer resources.

## Key APIs and Concepts

- **Frame Extraction**: Each decoded frame is saved as a separate file:
  - Frames are saved in raw YUV format (NV12, P016, YUV444, etc.).
  - Sequential file naming for easy frame identification.
  - Preserves original video quality without re-encoding.
  - Useful for frame-by-frame analysis and quality verification.

- **File Naming Convention**: Frames are saved with zero-padded sequential numbers:
  - Format: `frame_XXXX.yuv` where XXXX is the frame number.
  - Zero-padding ensures proper alphabetical sorting.
  - Frame number corresponds to decode order.

- **Output Format**: Frames are saved in raw YUV format:
  - Maintains the decoder's native output format (NV12, P016, etc.).
  - No color space conversion or format transformation.
  - Includes luma and chroma planes in planar or semi-planar layout.
  - File size depends on resolution and bit depth.

- **Decoder Configuration**: Standard decoder setup with:
  - `rocDecCreateDecoder()`: Initializes decoder with codec parameters.
  - Output surface memory type configured for frame retrieval.
  - Optional crop rectangle for region-of-interest extraction.

- **Frame Processing**: Each frame is processed individually:
  - `rocDecGetVideoFrame()`: Retrieves decoded frame data.
  - Frame data is copied from device memory to host memory.
  - Written to disk as raw binary data.
  - Frame is released back to decoder surface pool.

- **Use Cases**:
  - Creating image sequences from video files.
  - Frame-by-frame quality analysis and inspection.
  - Extracting specific frames for thumbnails or previews.
  - Debugging video decode issues.
  - Preparing training data for machine learning applications.

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
