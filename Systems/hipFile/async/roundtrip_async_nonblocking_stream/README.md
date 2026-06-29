# hipFile Async Roundtrip Non-Blocking Stream Example

## Description

This example demonstrates asynchronous GPU-direct I/O using hipFile on an explicit
non-blocking HIP stream created with `hipStreamNonBlocking`. Unlike the default stream used in
`roundtrip_async`, a non-blocking stream does not implicitly synchronize with the HIP legacy
default stream.

### Application flow

1. The GPU device is selected.
2. The input file `READ_FILE` is seeded on disk with a deterministic pattern via plain POSIX I/O.
3. A GPU buffer is allocated and registered with `hipFileBufRegister`.
4. Both files are opened with `O_DIRECT` and registered with hipFile.
5. A non-blocking stream is created with `hipStreamCreateWithFlags(hipStreamNonBlocking)`.
6. The GPU buffer is zeroed asynchronously on the non-blocking stream.
7. `hipFileReadAsync` is submitted on the non-blocking stream, followed immediately by
   `hipFileWriteAsync` on the **same** stream. Stream-local ordering guarantees the write
   starts after the read completes.
8. `hipStreamSynchronize` blocks until both operations complete.
9. `ftruncate` trims the output file to the logical payload size.
10. Both files are hashed and compared.
11. `hipStreamDestroy` releases the stream. All other resources are released in reverse order.

## Key APIs and Concepts

- `hipStreamCreateWithFlags(hipStreamNonBlocking)` creates a stream that does not synchronize
  with the NULL (legacy default) stream. Because it prevents unintended serialization, this is
  the preferred choice for async hipFile I/O in applications that also issue work on the default
  stream.
- `hipStreamNonBlocking` streams still provide intra-stream ordering guarantees: operations
  submitted to the same stream run in submission order.
- `hipStreamDestroy` must be called after all work on the stream has completed (confirmed by
  `hipStreamSynchronize`).

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
