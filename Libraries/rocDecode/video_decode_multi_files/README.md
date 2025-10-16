# rocDecode Multi-File Video Decode with Reconfiguration

## Description

This example demonstrates the decoder reconfiguration capability of the rocDecode library by decoding multiple video files with a single decoder instance. The sample showcases how to handle video files with different resolutions or parameters using the same decoder, leveraging the reconfigure feature to adapt to changing video properties without recreating the decoder. Input files must be of the same codec type but can have varying resolutions.

## Application Flow

1. Parse command-line arguments for input file list, device ID, and reconfiguration options.
2. Read the input file list containing paths to multiple video files.
3. Store video file paths in a queue for sequential processing.
4. Initialize the first video file's demuxer to extract codec information.
5. Create the decoder instance with the codec from the first file.
6. Set up reconfiguration callback for handling resolution changes.
7. For each video file in the queue:
   - Open the video file with the demuxer.
   - If resolution differs from previous file, trigger decoder reconfiguration.
   - Decode all frames from the current file.
   - Flush remaining frames when switching to the next file.
   - Optionally save decoded frames to separate output files.
8. Display decoding statistics for all processed files.
9. Clean up decoder and demuxer resources.

## Key APIs and Concepts

- **Decoder Reconfiguration**: The decoder can adapt to different video parameters without recreation:
  - Triggered automatically when the parser detects sequence parameter changes.
  - Handles resolution changes, bit depth changes, and chroma format changes.
  - Reconfiguration callback flushes pending frames before applying new parameters.
  - More efficient than destroying and recreating the decoder for each file.

- **Reconfiguration Callback**: A user-provided callback function is invoked during reconfiguration:
  - Called when video sequence parameters change between files.
  - Flushes any remaining decoded frames from the previous sequence.
  - Optionally saves flushed frames to output files.
  - Returns the number of frames flushed.
  - Allows the decoder to reset its internal state for the new sequence.

- **Multi-File Processing**: The sample processes multiple files sequentially:
  - Files are read from an input list file (one path per line).
  - Each file is demuxed and decoded independently.
  - The decoder maintains state across file boundaries.
  - Output can be saved to separate files per input video.

- **Codec Compatibility**: All input files must use the same codec:
  - Codec type (H.264, HEVC, VP9, etc.) must be consistent.
  - Resolution, frame rate, and bit depth can vary.
  - The decoder reconfigures for parameter changes but cannot switch codecs.

- **Flush Modes**: The reconfiguration callback supports different flush modes:
  - `RECONFIG_FLUSH_MODE_NONE`: Just count flushed frames.
  - `RECONFIG_FLUSH_MODE_DUMP_TO_FILE`: Save flushed frames to output file.
  - `RECONFIG_FLUSH_MODE_CALCULATE_MD5`: Generate MD5 digest for flushed frames.
  - Modes can be combined using bitwise OR.

- **Use Cases**:
  - Processing video playlists with varying resolutions.
  - Adaptive bitrate streaming scenarios.
  - Batch processing of related video files.
  - Testing decoder robustness with parameter changes.

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
- `avformat_close_input`
- `avformat_find_stream_info`
- `av_find_best_stream`
- `av_read_frame`
- `av_packet_alloc`
- `av_packet_free`
- `av_packet_unref`
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
