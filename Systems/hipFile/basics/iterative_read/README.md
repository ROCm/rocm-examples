# hipFile Iterative Read Example

## Description

This example demonstrates reading a file into GPU memory in fixed-size chunks by advancing the
destination pointer into the GPU buffer on each iteration, rather than using a fixed base
pointer with a varying `buf_offset`. After all chunks are read, the full GPU buffer is written to
an output file in a single call, and the two files are verified to match.

This is useful for streaming large files into GPU memory when only a portion of the file fits in
the GPU buffer at one time, or when the caller wants to process each chunk independently.

### Application flow

1. The GPU device is selected.
2. The input file is stat'd to determine its size and the filesystem block size.
3. The input file is opened with `O_DIRECT` and registered with hipFile.
4. A device buffer is allocated with `hipMalloc` (rounded up to the block size) and zeroed.
5. A loop calls `hipFileRead` with an advancing host pointer (`(char*)devbuf + bytes_read`) and
   a matching `file_offset`. Each call reads one chunk (default 64 KiB) or the remaining bytes,
   whichever is smaller. The loop terminates when all bytes are read or EOF is reached.
6. The output file is opened with `O_DIRECT` and registered. `hipFileWrite` writes the entire
   GPU buffer in one call.
7. `ftruncate` trims the output file to the logical payload size.
8. Both files are hashed via plain POSIX I/O and compared to verify correctness.
9. All resources are released.

## Key APIs and Concepts

- `hipFileRead` accepts a raw pointer into GPU memory, a transfer size, a `file_offset`, and a
  `buf_offset`. Advancing the raw pointer (`buf`) each iteration while keeping `buf_offset` at
  zero is the unregistered-buffer pattern for iterative reads. Advancing `buf_offset` with a
  fixed base pointer (see `iterative_devmem_offset_read`) is preferred for registered buffers.
- Chunk sizes must be multiples of the filesystem's logical block size when `O_DIRECT` is in use.
- `hipFileWrite` can write from a raw device pointer (no explicit buffer registration required),
  using the internal pool buffer if the pointer is unregistered.

## Demonstrated API Calls

### hipFile runtime

- `hipFileHandleRegister`
- `hipFileHandleDeregister`
- `hipFileRead`
- `hipFileWrite`

### HIP runtime

- `hipSetDevice`
- `hipMalloc`
- `hipFree`
- `hipMemset`
