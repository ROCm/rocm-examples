# rocDecode Raw Bitstream Decode

## Description

This example demonstrates video decoding using the rocDecode library's built-in bitstream reader instead of FFMPEG demuxer. The sample reads raw video bitstreams directly and decodes them using hardware acceleration, providing an alternative to FFMPEG-based demuxing. This approach is useful for scenarios where FFMPEG is not available or when working with raw elementary streams.

## Application Flow

1. Parse command-line arguments for input file, device ID, and output options.
2. Create the built-in bitstream reader using `rocDecCreateBitstreamReader()`.
3. Extract codec information from the bitstream.
4. Create the video decoder instance with the detected codec.
5. Verify codec support on the selected GPU device.
6. Loop through bitstream:
   - Read video data packets using the bitstream reader.
   - Parse packets using the video parser.
   - Decode frames through parser callbacks.
   - Retrieve decoded frames.
   - Optionally save frames to output file.
7. Display decoding statistics.
8. Destroy bitstream reader and clean up decoder resources.

## Key APIs and Concepts

- **Built-in Bitstream Reader**: rocDecode provides a native bitstream reader:
  - `rocDecCreateBitstreamReader()`: Creates a reader for raw elementary streams.
  - `rocDecDestroyBitstreamReader()`: Releases bitstream reader resources.
  - Eliminates dependency on FFMPEG for simple decode scenarios.
  - Supports common video codecs (H.264, HEVC, VP9, AV1).

- **Raw Elementary Stream Processing**: The bitstream reader handles:
  - Direct reading of compressed video data.
  - Automatic detection of codec type from bitstream.
  - Extraction of sequence parameters and frame data.
  - No container format parsing required.

- **Decoder Integration**: The decoder works with the bitstream reader:
  - `rocDecCreateDecoder()`: Initializes decoder with codec from bitstream.
  - `rocDecParseVideoData()`: Parses data provided by bitstream reader.
  - `rocDecDecodeFrame()`: Decodes frames using hardware acceleration.
  - Same decode pipeline as FFMPEG-based samples.

- **Use Cases**:
  - Decoding raw elementary streams without container formats.
  - Embedded systems where FFMPEG is not available.
  - Custom video processing pipelines.
  - Testing decoder with raw bitstream data.
  - Scenarios requiring minimal dependencies.

- **Limitations**:
  - Only supports elementary streams (no container formats like MP4, MKV).
  - Limited metadata extraction compared to FFMPEG.
  - No support for multiplexed audio/video streams.
  - Codec must be detectable from bitstream headers.

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
- `rocDecCreateBitstreamReader`
- `rocDecDestroyBitstreamReader`
- `rocDecGetErrorName`

### HIP Runtime APIs

- `hipSetDevice`
- `hipGetDeviceProperties`
- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyDtoH`

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
