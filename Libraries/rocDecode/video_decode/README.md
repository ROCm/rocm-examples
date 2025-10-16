# rocDecode Video Decode

## Description

This example illustrates the use of the rocDecode library for decoding a single packetized video stream using FFMPEG demuxer, video parser, and hardware-accelerated decoder to obtain individual decoded frames in YUV format. The sample demonstrates the standard video decoding workflow with configurable options for device selection, output file dumping, frame limiting, and MD5 validation. It uses a high-level wrapper class that integrates both the video parser and decoder for simplified usage.

## Application Flow

1. Parse command-line arguments for input file, output path, device ID, and decoding options.
2. Initialize the FFMPEG video demuxer to extract codec information and video packets.
3. Create the video decoder instance with specified codec, device, and output configuration.
4. Verify codec support on the selected GPU device.
5. Set up optional MD5 generator for frame validation.
6. Loop through video stream:
   - Demux video packets from input file.
   - Decode frames using the rocDecode API.
   - Retrieve decoded frames from the decoder.
   - Optionally save frames to output file.
   - Optionally generate MD5 digest for validation.
   - Release decoded frames back to the decoder.
7. Display decoding statistics including frame count and performance metrics.
8. Optionally compare generated MD5 digest with reference.
9. Clean up decoder and demuxer resources.

## Key APIs and Concepts

- **Video Demuxer**: Uses FFMPEG libraries to demux video files and extract codec parameters, frame rate, resolution, and compressed video packets. The demuxer supports various container formats (MP4, MKV, AVI, etc.) and provides packet-level access to the video stream.

- **Decoder Initialization**: The decoder is initialized with:
  - `rocDecCreateDecoder()`: Creates a decoder instance configured with codec type, output surface format, target dimensions, and memory type.
  - Configuration includes output surface memory type (device internal, device copied, host copied, or not mapped).
  - Optional crop rectangle for region-of-interest decoding.
  - Display delay parameter for controlling output latency.

- **Video Parser**:
  - `rocDecCreateVideoParser()`: Creates a parser that handles bitstream parsing and manages the decode pipeline through callbacks.
  - `rocDecParseVideoData()`: Parses compressed video data and triggers decode operations.
  - Parser callbacks handle sequence changes, picture decode, and picture display events.

- **Frame Decoding**:
  - `rocDecDecodeFrame()`: Decodes a frame using hardware acceleration. Called internally by the parser callback.
  - `rocDecGetDecodeStatus()`: Queries the decode status of a frame to ensure completion before retrieval.
  - Supports various output formats including NV12, P016, YUV444, and their 16-bit variants.

- **Frame Management**:
  - Decoded frames are retrieved using `rocDecGetVideoFrame()` which provides device memory pointers.
  - Frames must be explicitly released using the release mechanism to return surfaces to the decoder pool.
  - Supports configurable display delay for reordering frames from decode order to display order.

- **Output Options**:
  - Frames can be saved to file in raw YUV format.
  - MD5 digest generation for decoded frame validation.
  - SEI (Supplemental Enhancement Information) message extraction.
  - Configurable output surface memory types for different use cases.

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
- `av_rescale_q`
- `av_q2d`

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
- `RocdecSeiMessageInfo`
- `rocDecDecodeStatus`
- `AVCodecID`
- `AVFormatContext`
- `AVPacket`
