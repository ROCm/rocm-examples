# rocDecode Batch Video Decode

## Description

This example demonstrates batch video decoding using multiple threads with the rocDecode library. The sample decodes multiple video files concurrently, distributing the workload across multiple threads to maximize GPU utilization and throughput. It showcases efficient multi-threaded video decoding with configurable thread count and automatic load balancing across available files.

## Application Flow

1. Parse command-line arguments for input directory, number of threads, device ID, and output options.
2. Scan the input directory to collect all video files.
3. Determine the optimal number of threads based on file count and user request (maximum 64 threads).
4. Distribute video files across threads using round-robin assignment.
5. For each thread:
   - Initialize a separate video demuxer for assigned files.
   - Create a dedicated decoder instance.
   - Process assigned files sequentially within the thread.
   - Decode all frames from each file.
   - Optionally save decoded frames to output.
   - Generate MD5 digest for validation if requested.
6. Synchronize all threads and collect decoding statistics.
7. Display aggregate performance metrics including total frames decoded and throughput.
8. Clean up all decoder and demuxer resources.

## Key APIs and Concepts

- **Multi-Threading**: The sample creates multiple threads, each with its own decoder instance, to process video files in parallel. This approach maximizes GPU utilization by keeping the hardware decoder busy with multiple decode streams.

- **Thread-Safe Decoder Instances**: Each thread maintains its own:
  - `rocDecCreateDecoder()`: Creates an independent decoder instance per thread.
  - Video demuxer for reading input files.
  - Frame buffers and output resources.
  - This design avoids synchronization overhead and allows true parallel decoding.

- **Load Balancing**: Files are distributed across threads using round-robin assignment:
  - If files > threads: Multiple files per thread, distributed evenly.
  - If files < threads: One file per thread, unused threads are not created.
  - Ensures balanced workload across all active threads.

- **Decoder Configuration**: Each decoder is configured with:
  - Device ID for GPU selection.
  - Output surface memory type (typically device internal for performance).
  - Codec-specific parameters extracted from input files.
  - Display delay and surface pool size for optimal throughput.

- **Frame Processing Pipeline**: Within each thread:
  - `rocDecParseVideoData()`: Parses video packets from demuxer.
  - `rocDecDecodeFrame()`: Decodes frames using hardware acceleration.
  - `rocDecGetVideoFrame()`: Retrieves decoded frames.
  - Frames are processed and released to maintain decoder surface pool.

- **Performance Optimization**:
  - Parallel decoding across multiple streams.
  - Minimal synchronization between threads.
  - Efficient memory management with surface reuse.
  - Optional output to avoid I/O bottlenecks during performance testing.

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
- `hipGetDeviceCount`
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

### C++ Standard Library (Threading)

- `std::thread`
- `std::vector`
- `std::mutex`
- `std::chrono`

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
