# rocDecode Video Decode Capabilities

## Description

This example queries and prints the hardware video decode capabilities of an AMD GPU using the rocDecode library's `rocDecGetDecoderCaps` API. Hardware decode support depends on the GPU's Video Core Next (VCN) generation, so a codec that decodes on one GPU may not be available on another; for example, AV1 decoding is not supported on gfx90a (MI210). Applications can use this query to check whether a stream can be decoded before creating a decoder. Unlike the other rocDecode examples, this example does not decode a stream, so it needs no input file and does not depend on FFMPEG.

## Application Flow

1. Parse command-line arguments for the GPU device ID.
2. Query the device name, architecture, and PCI bus ID with HIP.
3. For every codec (MPEG1, MPEG2, MPEG4, AVC/H.264, HEVC/H.265, AV1, VP8, VP9), chroma format (monochrome, 4:2:0, 4:2:2, 4:4:4), and bit depth (8, 10, 12):
   - Fill a `RocdecDecodeCaps` structure with the device ID, codec, chroma format, and bit depth.
   - Call `rocDecGetDecoderCaps()` and skip the combination if `is_supported` is not set.
   - Record the number of decoders, the minimum and maximum coded resolution, and the supported output surface formats.
4. Merge bit depths with identical capabilities into one row and print a table per codec.
5. Print the total number of supported combinations.

## Key APIs and Concepts

- **Decoder Capability Query**: `rocDecGetDecoderCaps()` reports whether the GPU's hardware decoder supports a codec, chroma format, and bit depth combination:
  - Inputs: `device_id`, `codec_type`, `chroma_format`, and `bit_depth_minus_8`.
  - Outputs: `is_supported`, `num_decoders`, `min_width`/`min_height`, `max_width`/`max_height`, and `output_format_mask`.
  - The query returns an error status only if it fails; an unsupported combination is reported through `is_supported`.

- **Output Surface Formats**: Each bit in `output_format_mask` corresponds to a `rocDecVideoSurfaceFormat` value (for example NV12 for 8-bit 4:2:0 or P016 for 10-bit 4:2:0) that the decoder can produce for that combination.

- **Use Cases**:
  - Checking codec support before decoding, for example to skip AV1 streams on GPUs without AV1 decode.
  - Selecting a GPU that can decode a given stream on a multi-GPU system.
  - Validating stream resolutions against the decoder's limits.

## Demonstrated API Calls

### rocDecode APIs

- `rocDecGetDecoderCaps`
- `rocDecGetErrorName`

### HIP Runtime APIs

- `hipGetDeviceCount`
- `hipSetDevice`
- `hipGetDeviceProperties`
- `hipDeviceGetPCIBusId`

### Data Types and Enums

- `RocdecDecodeCaps`
- `rocDecVideoCodec`
- `rocDecVideoChromaFormat`
- `rocDecVideoSurfaceFormat`
- `rocDecStatus`
