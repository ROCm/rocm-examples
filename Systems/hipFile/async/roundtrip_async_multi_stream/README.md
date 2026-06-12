# hipFile Async Multi-Stream Roundtrip Example

## Description

This program demonstrates concurrent GPU-direct I/O using multiple independent HIP streams. Each
stream processes a non-overlapping slice of the file, with its own GPU buffer. A read and a write
are submitted on each stream, allowing the driver to execute slices in parallel. After all streams
synchronize, the complete file (all slices concatenated) is verified against the input.

### Application flow

1. The GPU device is selected.
2. The input file `READ_FILE` is seeded with `NUM_STREAMS × SLICE_SIZE` bytes of deterministic
   pattern via plain POSIX I/O.
3. For each of `NUM_STREAMS` (default 4) streams:
   - A `SLICE_SIZE` (default 1 MiB) GPU buffer is allocated and registered.
   - A non-blocking stream is created. The buffer is zeroed on it.
   - The stream's file offset is set to `i × SLICE_SIZE`.
4. Both files are opened with `O_DIRECT` and registered. A single `hipFileHandle_t` pair is
   shared across all streams.
5. `hipFileReadAsync` is submitted for each stream.
6. `hipFileWriteAsync` is submitted for each stream. Per-stream ordering ensures each write sees
   its stream's read result.
7. Each stream is synchronized and its byte counts are verified.
8. Both files are hashed across the full `TOTAL_SIZE` and compared.
9. Per-entry cleanup: stream destroy, buffer deregister, free.

## Key APIs and Concepts

- Multiple non-blocking streams allow hipFile operations from different slices to execute
  concurrently in the driver, potentially overlapping I/O with computation or other I/O.
- A shared `hipFileHandle_t` is safe to use across multiple concurrent streams for non-overlapping
  byte ranges. Each stream owns its own GPU buffer.
- The `io_size`, `file_offset`, `buf_offset`, `bytes_read`, and `bytes_written` fields live in
  per-stream state structs because the async API takes them by pointer and may write to them at
  completion time, so they must outlive the `hipStreamSynchronize` call.
- `SLICE_SIZE` must be a multiple of `BLOCK_ALIGN` (4 KiB) because the files are opened with
  `O_DIRECT`.

## Demonstrated API Calls

### hipFile runtime

- `hipFileBufRegister`
- `hipFileBufDeregister`
- `hipFileHandleRegister`
- `hipFileHandleDeregister`
- `hipFileReadAsync`
- `hipFileWriteAsync`

### HIP runtime

- `hipSetDevice`
- `hipMalloc`
- `hipFree`
- `hipStreamCreateWithFlags`
- `hipStreamNonBlocking`
- `hipMemsetAsync`
- `hipStreamSynchronize`
- `hipStreamDestroy`
