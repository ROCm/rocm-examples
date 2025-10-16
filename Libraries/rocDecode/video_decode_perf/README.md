# rocDecode Performance Testing

## Description

This example demonstrates performance testing and benchmarking of the rocDecode library by decoding the same video file multiple times in parallel using multiple threads. The sample is designed to measure maximum decode throughput, GPU utilization, and multi-stream decode performance. It provides detailed performance metrics including frames per second, decode time per frame, and overall throughput.

## Application Flow

1. Parse command-line arguments for input file, number of threads, device ID, and performance options.
2. Initialize the video demuxer to extract codec information.
3. Verify codec support on the selected GPU device.
4. Create multiple threads (default or user-specified count).
5. For each thread:
   - Create an independent decoder instance.
   - Create a separate demuxer for the same input file.
   - Decode all frames from the video file.
   - Track decode time and frame count.
   - Optionally skip frame retrieval to measure pure decode performance.
6. Synchronize all threads and collect performance statistics.
7. Calculate and display aggregate metrics:
   - Total frames decoded across all threads.
   - Average decode time per frame.
   - Frames per second (FPS) per thread and aggregate.
   - Total throughput in FPS.
8. Clean up all decoder and demuxer resources.

## Key APIs and Concepts

- **Parallel Decode Streams**: Multiple decoder instances run concurrently:
  - Each thread has its own decoder created with `rocDecCreateDecoder()`.
  - Threads decode the same video file independently.
  - Maximizes GPU hardware decoder utilization.
  - Tests multi-stream decode capability of the hardware.

- **Performance Measurement**: The sample tracks detailed timing information:
  - Decode start and end times for each thread.
  - Per-frame decode time.
  - Total decode time excluding initialization overhead.
  - Frame count per thread and aggregate.

- **Decode-Only Mode**: Optional mode to measure pure decode performance:
  - Skips frame retrieval and memory copies.
  - Focuses on hardware decode throughput.
  - Useful for understanding decoder bottlenecks vs. memory bandwidth.

- **Thread Configuration**: Configurable number of parallel decode threads:
  - Default thread count based on system capabilities.
  - User can specify thread count for testing different scenarios.
  - Each thread operates independently without synchronization during decode.

- **GPU Utilization**: The sample helps measure:
  - Maximum decode throughput of the GPU.
  - Scalability with multiple concurrent streams.
  - Hardware decoder saturation point.
  - Efficiency of parallel decode operations.

- **Performance Metrics**:
  - **Per-Thread FPS**: Decode rate for individual threads.
  - **Aggregate FPS**: Combined throughput across all threads.
  - **Average Decode Time**: Mean time to decode a single frame.
  - **Total Frames**: Sum of frames decoded by all threads.

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

### C++ Standard Library (Threading and Timing)

- `std::thread`
- `std::vector`
- `std::chrono::high_resolution_clock`
- `std::chrono::duration`

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
