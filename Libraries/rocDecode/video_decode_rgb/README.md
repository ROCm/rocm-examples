# rocDecode Video Decode with Color Space Conversion

## Description

This example demonstrates video decoding with hardware-accelerated color space conversion using the rocDecode library and custom HIP kernels. The sample decodes YUV video frames and converts them to RGB/BGR formats using GPU compute, showcasing parallel execution of VCN hardware decoder and compute engine. It supports multiple output formats including RGB24, BGR24, RGBA32, BGRA32, and their 48/64-bit variants, with optional frame resizing.

## Application Flow

1. Parse command-line arguments for input file, output format, resize dimensions, and device options.
2. Initialize the FFMPEG video demuxer to extract codec information.
3. Create the video decoder instance with device internal memory type.
4. Verify codec support on the selected GPU device.
5. Create two HIP streams: one for decoding, one for color space conversion.
6. Allocate frame buffers for asynchronous processing.
7. Launch color space conversion thread for parallel post-processing.
8. Loop through video stream:
   - Demux video packets from input file.
   - Decode frames using hardware decoder.
   - Copy decoded frames to intermediate buffers asynchronously.
   - Signal color space conversion thread when frames are ready.
   - Conversion thread processes frames in parallel:
     - Optional resize using HIP kernels.
     - Color space conversion from YUV to RGB/BGR.
     - Save converted frames to output file.
9. Synchronize threads and display performance metrics.
10. Clean up streams, buffers, and decoder resources.

## Key APIs and Concepts

- **Dual-Stream Processing**: Uses separate HIP streams for decode and post-processing:
  - Decode stream handles frame decoding and memory copies.
  - Color conversion stream handles YUV to RGB transformation.
  - Enables parallel execution of VCN decoder and compute engine.
  - Maximizes GPU utilization and throughput.

- **Color Space Conversion Kernels**: Custom HIP kernels for YUV to RGB conversion:
  - Supports multiple input formats: NV12, P016, YUV444, YUV444P16.
  - Supports multiple output formats: RGB24, BGR24, RGB48, BGR48, RGBA32, BGRA32, RGBA64, BGRA64.
  - Implements ITU-R BT.709 color space standard.
  - Optimized for AMD GPU architecture.

- **Resize Kernels**: Optional frame resizing using HIP kernels:
  - `ResizeNv12()`: Resize NV12 format frames.
  - `ResizeP016()`: Resize P016 format frames.
  - Nearest neighbor interpolation for performance.
  - Maintains aspect ratio or custom dimensions.

- **Asynchronous Processing**: Frame buffers enable pipelined execution:
  - Decoded frames are copied to intermediate buffers.
  - Conversion thread processes frames while decoder continues.
  - Condition variables synchronize producer-consumer pattern.
  - Minimizes idle time for both decoder and compute.

- **Output Formats**:
  - **24-bit**: RGB24, BGR24 (8-bit per channel, 3 channels).
  - **32-bit**: RGBA32, BGRA32 (8-bit per channel, 4 channels with alpha).
  - **48-bit**: RGB48, BGR48 (16-bit per channel, 3 channels).
  - **64-bit**: RGBA64, BGRA64 (16-bit per channel, 4 channels with alpha).

- **Performance Optimization**:
  - Parallel decode and color conversion.
  - Asynchronous memory operations.
  - Efficient kernel implementations.
  - Optional output to avoid I/O bottlenecks.

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
- `hipMemcpyDtoDAsync`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`

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

### C++ Standard Library (Threading)

- `std::thread`
- `std::mutex`
- `std::condition_variable`
- `std::atomic`
- `std::queue`

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
- `hipStream_t`
- `AVCodecID`
- `AVFormatContext`
- `AVPacket`
