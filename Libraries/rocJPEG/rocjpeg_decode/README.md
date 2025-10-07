# rocJPEG JPEG Decode

## Description

This example illustrates the use of the `rocJPEG` library for decoding JPEG images on AMD GPUs using hardware-accelerated VCN (Video Core Next) decoders.

The sample demonstrates:

- Parsing JPEG image streams
- Retrieving image metadata (dimensions, chroma subsampling)
- Decoding JPEG images to various output formats
- Optional region-of-interest (ROI) cropping
- Saving decoded images to disk

## Application flow

1. Parse command-line arguments for input path, output path, device ID, backend, output format, and crop rectangle.
2. Validate and configure decode parameters including output format and optional crop rectangle.
3. Discover JPEG files from the input path (single file or directory).
4. Initialize HIP device with the specified device ID.
5. Create rocJPEG handle with the selected backend (hardware-accelerated).
6. Create rocJPEG stream handle for parsing JPEG data.
7. For each JPEG file:
   - Read the JPEG file data from disk into memory.
   - Parse the JPEG stream to extract structure and metadata.
   - Retrieve image information (dimensions, number of components, chroma subsampling).
   - Validate image resolution and chroma subsampling support.
   - Calculate channel pitch and sizes based on output format and subsampling.
   - Allocate device memory for output channels (reusing buffers when possible).
   - Decode the JPEG image to the specified output format.
   - Optionally save the decoded image to disk.
   - Calculate and display performance metrics.
8. Free allocated device memory for all channels.
9. Destroy rocJPEG stream handle and rocJPEG handle.
10. Display summary statistics for batch processing.

## Key APIs and Concepts

- **rocJPEG Initialization**: The rocJPEG library is initialized by creating a handle with `rocJpegCreate()` specifying the backend type and device ID. The handle is released with `rocJpegDestroy()`.

- **Stream Handling**:
  - `rocJpegStreamCreate()`: Creates a stream handle for parsing and decoding JPEG data.
  - `rocJpegStreamParse()`: Parses a JPEG bitstream to extract metadata and prepare for decoding.
  - `rocJpegStreamDestroy()`: Frees the stream handle.

- **Image Information**:
  - `rocJpegGetImageInfo()`: Retrieves image metadata including number of components, chroma subsampling format, and dimensions for each component.

- **Decoding**:
  - `rocJpegDecode()`: Decodes a single JPEG image using the parsed stream, decode parameters, and output image structure.

- **Decode Parameters** (`RocJpegDecodeParams`):
  - `output_format`: Specifies the desired output format (native, YUV planar, Y-only, RGB, RGB planar).
  - `crop_rectangle`: Optional region-of-interest with left, top, right, and bottom coordinates for cropping during decode.

- **Output Image Structure** (`RocJpegImage`):
  - `channel[]`: Array of device memory pointers for each output channel (up to 3 channels).
  - `pitch[]`: Array of pitch values (stride in bytes) for each channel.

- **Chroma Subsampling Formats**:
  - `ROCJPEG_CSS_444`: YUV 4:4:4 (no subsampling)
  - `ROCJPEG_CSS_440`: YUV 4:4:0 (horizontal full, vertical half)
  - `ROCJPEG_CSS_422`: YUV 4:2:2 (horizontal half, vertical full)
  - `ROCJPEG_CSS_420`: YUV 4:2:0 (half resolution for both chroma channels)
  - `ROCJPEG_CSS_400`: YUV 4:0:0 (grayscale, no chroma)
  - `ROCJPEG_CSS_411`: YUV 4:1:1 (not supported by VCN hardware)
  - `ROCJPEG_CSS_UNKNOWN`: Unknown subsampling (not supported)

- **Output Formats**:
  - `ROCJPEG_OUTPUT_NATIVE`: Native format based on chroma subsampling (e.g., NV12 for 4:2:0, YUYV for 4:2:2)
  - `ROCJPEG_OUTPUT_YUV_PLANAR`: Planar YUV format with separate Y, U, V channels
  - `ROCJPEG_OUTPUT_Y`: Y-channel only (grayscale)
  - `ROCJPEG_OUTPUT_RGB`: Interleaved RGB format
  - `ROCJPEG_OUTPUT_RGB_PLANAR`: Planar RGB format with separate R, G, B channels

- **Backend Types**:
  - `ROCJPEG_BACKEND_HARDWARE`: Hardware-accelerated decoding using VCN
  - `ROCJPEG_BACKEND_HYBRID`: Hybrid mode (currently not supported)

- **Error Handling**:
  - `rocJpegGetErrorName()`: Returns a string description of a rocJPEG status code.
  - Status codes include `ROCJPEG_STATUS_SUCCESS` and various error conditions.

## Demonstrated API Calls

### rocJPEG

- `rocJpegCreate`
- `rocJpegDestroy`
- `rocJpegStreamCreate`
- `rocJpegStreamParse`
- `rocJpegStreamDestroy`
- `rocJpegGetImageInfo`
- `rocJpegDecode`
- `rocJpegGetErrorName`

### HIP runtime

- `hipGetDeviceCount`
- `hipSetDevice`
- `hipGetDeviceProperties`
- `hipMalloc`
- `hipFree`
- `hipMemcpyDtoH`

### Data Types and Enums

- `RocJpegHandle`
- `RocJpegStreamHandle`
- `RocJpegStatus`
- `RocJpegBackend`
- `RocJpegDecodeParams`
- `RocJpegImage`
- `RocJpegChromaSubsampling`
- `RocJpegOutputFormat`
- `ROCJPEG_STATUS_SUCCESS`
- `ROCJPEG_BACKEND_HARDWARE`
- `ROCJPEG_BACKEND_HYBRID`
- `ROCJPEG_CSS_444`
- `ROCJPEG_CSS_440`
- `ROCJPEG_CSS_422`
- `ROCJPEG_CSS_420`
- `ROCJPEG_CSS_400`
- `ROCJPEG_CSS_411`
- `ROCJPEG_CSS_UNKNOWN`
- `ROCJPEG_OUTPUT_NATIVE`
- `ROCJPEG_OUTPUT_YUV_PLANAR`
- `ROCJPEG_OUTPUT_Y`
- `ROCJPEG_OUTPUT_RGB`
- `ROCJPEG_OUTPUT_RGB_PLANAR`
- `ROCJPEG_MAX_COMPONENT`
