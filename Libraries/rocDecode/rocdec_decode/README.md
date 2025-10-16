# rocDecode Low-Level API Example

## Description

This example demonstrates the use of low-level rocDecode APIs for hardware-accelerated video decoding on AMD GPUs. It showcases both device-based and host-based decoding backends, providing direct control over the decoder initialization, frame decoding, and output retrieval. This sample is ideal for understanding the fundamental rocDecode API workflow without high-level wrapper abstractions.

## Application Flow

1. Parse command-line arguments for input file, device ID, and backend selection.
2. Initialize the video demuxer to extract codec information and video packets.
3. Set up the video parser with callback functions for sequence, decode, and display events.
4. Create the decoder instance based on the selected backend (device or host).
5. Configure decoder parameters including output surface format and dimensions.
6. Loop through video packets:
   - Parse video data using the video parser.
   - Decode frames through parser callbacks.
   - Retrieve decoded frames via display callbacks.
7. Extract decoded frame data to host or device memory based on backend.
8. Optionally save decoded frames to output file.
9. Clean up parser and decoder resources.

## Key APIs and Concepts

- **Decoder Initialization**: The rocDecode decoder is initialized using either `rocDecCreateDecoder()` for device-based decoding or `rocDecCreateDecoderHost()` for host-based decoding. The decoder configuration includes codec type, output surface format, dimensions, and number of decode surfaces.

- **Video Parser**:
  - `rocDecCreateVideoParser()`: Creates a parser instance that handles bitstream parsing and triggers callbacks for sequence changes, picture decode, and picture display events.
  - `rocDecParseVideoData()`: Parses video packet data and invokes registered callbacks to drive the decode process.
  - `rocDecDestroyVideoParser()`: Releases parser resources.

- **Frame Decoding**:
  - `rocDecDecodeFrame()` / `rocDecDecodeFrameHost()`: Decodes a single frame using the provided picture parameters. Called from the picture decode callback.
  - Picture parameters include current picture index, bitstream data, and decode-specific information.

- **Frame Retrieval**:
  - `rocDecGetVideoFrame()`: Retrieves decoded frame from device memory (device backend).
  - `rocDecGetVideoFrameHost()`: Retrieves decoded frame to host memory (host backend).
  - Both functions provide frame data, pitch information, and surface parameters.

- **Decoder Cleanup**:
  - `rocDecDestroyDecoder()`: Destroys device-based decoder instance.
  - `rocDecDestroyDecoderHost()`: Destroys host-based decoder instance.

- **Callback Functions**: The parser uses three callback functions:
  - **Sequence Callback**: Invoked when video sequence parameters are detected, used to initialize or reconfigure the decoder.
  - **Picture Decode Callback**: Called when a picture is ready to be decoded, triggers `rocDecDecodeFrame()`.
  - **Picture Display Callback**: Invoked when a decoded frame is ready for display, retrieves frame data using `rocDecGetVideoFrame()`.

## Demonstrated API Calls

### rocDecode APIs

- `rocDecCreateDecoder`
- `rocDecCreateDecoderHost`
- `rocDecDecodeFrame`
- `rocDecDecodeFrameHost`
- `rocDecGetVideoFrame`
- `rocDecGetVideoFrameHost`
- `rocDecDestroyDecoder`
- `rocDecDestroyDecoderHost`
- `rocDecCreateVideoParser`
- `rocDecParseVideoData`
- `rocDecDestroyVideoParser`
- `rocDecGetErrorName`

### HIP Runtime APIs

- `hipGetDeviceCount`
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
