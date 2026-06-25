# hipFile Async Multi-Stream Registered Roundtrip Example

## Description

This program extends the `roundtrip_async_multi_stream` example by additionally registering each
HIP stream with `hipFileStreamRegister`. Stream registration allows the hipFile driver to skip
per-submission validation of fixed parameters (buffer offset, file offset, file size, and
alignment), reducing submission overhead on latency-sensitive workloads.

### Application flow

1. The GPU device is selected.
2. The input file `READ_FILE` is seeded with `NUM_STREAMS × SLICE_SIZE` bytes of deterministic
   pattern via plain POSIX I/O.
3. For each of `NUM_STREAMS` (default 4) streams:
   - A `SLICE_SIZE` (default 1 MiB) GPU buffer is allocated and registered with
     `hipFileBufRegister`.
   - A non-blocking stream is created. The buffer is zeroed on it.
   - `hipFileStreamRegister` is called with the combined flags
     `HIPFILE_STREAM_FIXED_BUF_OFFSET | HIPFILE_STREAM_FIXED_FILE_OFFSET |
     HIPFILE_STREAM_FIXED_FILE_SIZE | HIPFILE_STREAM_PAGE_ALIGNED_INPUTS`, informing the driver
     that these parameters will not change across submissions on this stream.
4. Both files are opened with `O_DIRECT` and registered.
5. Reads, writes, and synchronization proceed identically to `roundtrip_async_multi_stream`.
6. Cleanup: `hipFileStreamDeregister` before `hipStreamDestroy`, then buffer deregister and free.

## Key APIs and Concepts

- `hipFileStreamRegister` registers a HIP stream with hipFile and attaches optimization hints.
  The driver can use these hints to cache and skip validation work that would otherwise repeat on
  every `hipFileReadAsync` / `hipFileWriteAsync` call that uses this stream.
- `HIPFILE_STREAM_FIXED_BUF_OFFSET` asserts that `buf_offset` will be the same value on every
  submission to this stream.
- `HIPFILE_STREAM_FIXED_FILE_OFFSET` asserts the same for `file_offset`.
- `HIPFILE_STREAM_FIXED_FILE_SIZE` asserts that the transfer size will not change.
- `HIPFILE_STREAM_PAGE_ALIGNED_INPUTS` asserts that all three values are page (4 KiB) aligned.
- `hipFileStreamDeregister` must be called before `hipStreamDestroy`. The tear-down order is:
  `hipFileStreamDeregister` → `hipStreamDestroy` → `hipFileBufDeregister` → `hipFree`.

## Demonstrated API Calls

### hipFile runtime

- `hipFileBufRegister`
- `hipFileBufDeregister`
- `hipFileHandleRegister`
- `hipFileHandleDeregister`
- `hipFileStreamRegister`
- `hipFileStreamDeregister`
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
