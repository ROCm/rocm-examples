# rocDecode Memory-Based Video Decode

## Description

This example demonstrates memory-based video decoding using the rocDecode library with a custom stream provider. Instead of reading directly from a file, the sample shows how to pass video data chunk-by-chunk sequentially to the FFMPEG demuxer, which is then decoded on AMD hardware. This approach is useful for scenarios where video data comes from network streams, memory buffers, or other non-file sources.

## Application Flow

1. Parse command-line arguments for input file path, device ID, and output options.
2. Create a custom `FileStreamProvider` class that implements the stream provider interface.
3. Initialize the video demuxer with the custom stream provider instead of a file path.
4. The stream provider reads the video file in chunks and fills the demuxer's buffer.
5. Create the video decoder instance with codec information from the demuxer.
6. Loop through the video stream:
   - The demuxer requests data from the stream provider as needed.
   - Stream provider reads chunks from the file into the demuxer's buffer.
   - Demuxer extracts video packets from the buffered data.
   - Decoder processes packets and produces decoded frames.
   - Retrieve and optionally save decoded frames.
7. Display decoding statistics and performance metrics.
8. Clean up decoder, demuxer, and stream provider resources.

## Key APIs and Concepts

- **Custom Stream Provider**: The sample implements a `FileStreamProvider` class derived from `VideoDemuxer::StreamProvider`:
  - `GetData()`: Called by the demuxer to fill its buffer with video data.
  - `GetBufferSize()`: Returns the size of the buffer to allocate.
  - This abstraction allows feeding video data from any source (file, network, memory, etc.).

- **Memory-Based Demuxing**: The FFMPEG demuxer is configured to use a custom I/O context:
  - `avio_alloc_context()`: Creates a custom I/O context with the stream provider.
  - The demuxer reads data through callbacks instead of direct file access.
  - Enables streaming scenarios where data arrives incrementally.

- **Decoder Integration**: The decoder works seamlessly with the memory-based demuxer:
  - `rocDecCreateDecoder()`: Initializes the decoder with codec parameters from the demuxer.
  - `rocDecParseVideoData()`: Parses video packets provided by the memory-based demuxer.
  - `rocDecDecodeFrame()`: Decodes frames using hardware acceleration.
  - No changes needed in the decode pipeline compared to file-based decoding.

- **Buffer Management**: The stream provider manages data buffering:
  - Reads video data in configurable chunk sizes.
  - Maintains read position and handles end-of-stream conditions.
  - Provides data to the demuxer on demand without loading the entire file into memory.

- **Use Cases**:
  - Network streaming applications where video data arrives over the network.
  - Processing video data from memory buffers or databases.
  - Implementing custom data sources with encryption or compression.
  - Real-time video processing pipelines with non-file inputs.

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

- `avformat_alloc_context`
- `avio_alloc_context`
- `avformat_open_input`
- `avformat_find_stream_info`
- `av_find_best_stream`
- `av_read_frame`
- `av_packet_alloc`
- `av_packet_free`
- `av_packet_unref`
- `avformat_close_input`
- `av_malloc`
- `av_freep`
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
- `AVFormatContext`
- `AVIOContext`
- `AVPacket`
- `AVCodecID`
