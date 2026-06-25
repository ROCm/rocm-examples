# hipFile Async Roundtrip Example

## Description

This program demonstrates asynchronous GPU-direct I/O using hipFile on the HIP default stream
(stream 0). A read and a write are both submitted asynchronously on the same stream; HIP
guarantees that operations on the same stream execute in submission order, so the write sees
the data deposited by the read without an intermediate explicit synchronization.

### Application flow

1. The GPU device is selected.
2. The input file `READ_FILE` is seeded on disk with a deterministic pattern via plain POSIX I/O.
   The seed size is rounded up to the block boundary so the `O_DIRECT` aligned read has valid
   content across its full transfer size.
3. A GPU buffer is allocated and zeroed asynchronously on the default stream.
4. The GPU buffer is registered with `hipFileBufRegister`.
5. Both files are opened with `O_DIRECT` and registered with hipFile.
6. `hipFileReadAsync` is submitted on the default stream (stream 0). `hipFileWriteAsync` is
   submitted on the same stream immediately after. Because both operations are on the same stream,
   the write is guaranteed to execute after the read completes.
7. `hipStreamSynchronize(0)` blocks until both operations complete. The byte counts are checked.
8. `ftruncate` trims the output file to the logical payload size.
9. Both files are hashed via plain POSIX I/O and compared.
10. All resources are released.

## Key APIs and Concepts

- `hipFileReadAsync` / `hipFileWriteAsync` submit non-blocking I/O operations onto a HIP stream.
  The operations complete asynchronously with respect to the host; call `hipStreamSynchronize`
  to wait for completion.
- The `size`, `file_offset`, and `buf_offset` parameters to the async calls are taken **by
  pointer** because the driver may update them at completion time. These values must stay valid
  (in scope and unmodified) until `hipStreamSynchronize` returns.
- Submitting a read and a write on the **same stream** provides implicit ordering — the write
  will not begin until the read has finished. Submitting them on different streams would require
  an explicit synchronization event between them.
- `hipFileBufRegister` is called before the async operations to ensure the GPU-direct path is
  used for both.

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
- `hipMemsetAsync`
- `hipStreamSynchronize`
