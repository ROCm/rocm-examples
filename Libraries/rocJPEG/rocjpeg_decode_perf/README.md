# rocJPEG JPEG Decode Performance

## Description

This example illustrates the use of the `rocJPEG` library for high-performance parallel decoding of JPEG images on AMD GPUs using hardware-accelerated VCN (Video Core Next) decoders with multi-threading support.

The sample demonstrates:

- Multi-threaded parallel JPEG decoding for maximum throughput
- Batch decoding of JPEG images with configurable batch size
- Thread pool management for concurrent decode operations
- Parsing multiple JPEG image streams across threads
- Retrieving image metadata for multiple images
- Decoding JPEG images to various output formats
- Optional region-of-interest (ROI) cropping
- Efficient memory management for parallel batch processing
- Performance measurement and aggregation across threads
- Saving decoded images to disk

## Application flow

1. Parse command-line arguments for input path, output path, device ID, backend, output format, number of threads, batch size, and crop rectangle.
2. Validate thread count (1-32 threads) and configure decode parameters including output format and optional crop rectangle.
3. Discover JPEG files from the input path (single file or directory).
4. Initialize HIP device with the specified device ID.
5. Adjust thread count based on the number of available images.
6. For each thread, initialize decode information structure:
   - Create rocJPEG handle with the selected backend (hardware-accelerated).
   - Create multiple rocJPEG stream handles (one per batch slot).
   - Initialize performance counters and error tracking.
7. Create thread pool with the specified number of worker threads.
8. Distribute JPEG files evenly among threads.
9. Submit decode jobs to the thread pool:
   - Each thread processes its assigned subset of images in batches.
   - For each batch within a thread:
     - Read JPEG file data from disk into memory.
     - Parse JPEG streams to extract structure and metadata.
     - Retrieve image information (dimensions, number of components, chroma subsampling).
     - Validate image resolution and chroma subsampling support.
     - Calculate channel pitch and sizes based on output format and subsampling.
     - Allocate device memory for output channels (reusing buffers when possible).
     - Decode the entire batch using `rocJpegDecodeBatched()`.
     - Calculate performance metrics for the batch.
     - Optionally save all decoded images from the batch to disk.
10. Wait for all threads to complete their decode jobs.
11. Aggregate performance statistics from all threads:
    - Total decoded images
    - Total images per second
    - Total megapixels per second
    - Skipped images by category
12. Free allocated device memory for all output channels across all threads.
13. Destroy all rocJPEG stream handles and rocJPEG handles for each thread.
14. Display comprehensive performance summary.

## Key APIs and Concepts

- **rocJPEG Initialization**: The rocJPEG library is initialized by creating a handle with `rocJpegCreate()` specifying the backend type and device ID. Multiple handles are created (one per thread) for parallel processing. Each handle is released with `rocJpegDestroy()`.

- **Stream Handling**:
  - `rocJpegStreamCreate()`: Creates a stream handle for parsing and decoding JPEG data. Multiple stream handles are created per thread for batch processing.
  - `rocJpegStreamParse()`: Parses a JPEG bitstream to extract metadata and prepare for decoding.
  - `rocJpegStreamDestroy()`: Frees the stream handle.

- **Image Information**:
  - `rocJpegGetImageInfo()`: Retrieves image metadata including number of components, chroma subsampling format, and dimensions for each component.

- **Batch Decoding**:
  - `rocJpegDecodeBatched()`: Decodes multiple JPEG images in a single batch operation. Takes an array of stream handles, batch size, decode parameters array, and output images array.
  - Batch processing improves throughput by submitting multiple decode operations together.
  - Each image in the batch can have different dimensions and chroma subsampling formats.

- **Multi-Threading Architecture**:
  - Thread pool manages worker threads for parallel decode operations.
  - Each thread has its own rocJPEG handle and stream handles to avoid contention.
  - Files are distributed evenly among threads for load balancing.
  - Each thread independently processes its assigned images in batches.
  - Performance metrics are tracked per thread and aggregated at the end.
  - `hipSetDevice()` is called within each thread to ensure proper device context.

- **Decode Parameters** (`RocJpegDecodeParams`):
  - `output_format`: Specifies the desired output format (native, YUV planar, Y-only, RGB, RGB planar).
  - `crop_rectangle`: Optional region-of-interest with left, top, right, and bottom coordinates for cropping during decode.
  - An array of decode parameters is used for batch processing, allowing per-image configuration.

- **Output Image Structure** (`RocJpegImage`):
  - `channel[]`: Array of device memory pointers for each output channel (up to 3 channels).
  - `pitch[]`: Array of pitch values (stride in bytes) for each channel.
  - An array of output image structures is used for batch processing.

- **Performance Optimization**:
  - Configurable number of threads (1-32) for parallel processing.
  - Configurable batch size for optimal GPU utilization.
  - Memory buffers are reused across batches when image sizes remain consistent.
  - Thread pool eliminates thread creation overhead.
  - Performance metrics include images per second and megapixels per second.

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
  - Per-thread error tracking for skipped images by category.

## Demonstrated API Calls

### rocJPEG

- `rocJpegCreate`
- `rocJpegDestroy`
- `rocJpegStreamCreate`
- `rocJpegStreamParse`
- `rocJpegStreamDestroy`
- `rocJpegGetImageInfo`
- `rocJpegDecodeBatched`
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
