# hipFile Batch Round-Trip Example

## Description

This example seeds an input file with a known pattern, reads it into a registered GPU buffer
using the hipFile batch I/O APIs, writes the buffer out to a second file in a second batch
phase, and verifies both files match by FNV-1a hash. Each phase keeps a continuously refilled
batch of 4 KiB requests in flight, so the queue stays full from the first submission to the last
completion. This is the batch API counterpart of the `basics/roundtrip_verify` example.

### Application flow

1. The GPU device is selected.
2. `READ_FILE` is created/truncated and seeded with the deterministic pattern via
   `seed_read_file`, and a `stat` call confirms the file now has the requested size.
3. A GPU buffer large enough to hold the whole file rounded up to a 4 KiB boundary
   (`BLOCK_ALIGN`, via `align_up`) is allocated with `hipMalloc`, zeroed with `hipMemset`, and
   synchronized with `hipDeviceSynchronize`. The buffer is registered with `hipFileBufRegister`.
4. `READ_FILE` is opened with `O_RDONLY | O_DIRECT`, and `WRITE_FILE` is opened with
   `O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT`, both via the `open_file` helper.
5. `hipFileBatchIOSetUp` creates a batch handle with capacity `BATCH_SIZE` (1 to 128; the
   largest capacity the API accepts is 128).
6. **Phase 1 – Read:** `hipFileBatchIOSubmit` submits up to `BATCH_SIZE` read requests covering
   the file in 4 KiB chunks, and `hipFileBatchIOGetStatus` (minimum 1) waits for completions.
   Each returned event is matched to its `BatchCookie` (which records the chunk index and the
   expected byte count) and validated: `status` must be `hipFileComplete` and `ret` must equal
   the expected size, so a short transfer is reported instead of failing silently. One
   replacement request is submitted for each returned event until all chunks have completed.
7. **Phase 2 – Write:** the same submit/status/refill loop runs with the write opcode against
   `WRITE_FILE`, transferring the file back out of the same GPU buffer. All reads finish before
   any writes are submitted.
8. `ftruncate` trims the final padded write to `FILE_SIZE`, and both files are closed with
   `close_file`.
9. `verify_files_match` hashes both files with FNV-1a and compares them. A match confirms the
   entire batched write-read-write cycle was lossless.
10. All resources are released: the batch handle is destroyed with `hipFileBatchIODestroy`, the
    GPU buffer is deregistered with `hipFileBufDeregister`, and memory is freed with `hipFree`.

## Key APIs and Concepts

- `hipFileBatchIOSetUp` / `hipFileBatchIODestroy` create and destroy a batch I/O context whose
  capacity is the maximum number of requests kept in flight. `BATCH_SIZE` can be 1 to 128; 128
  (`MAX_BATCH_SIZE`) is the largest capacity `hipFileBatchIOSetUp` accepts.
- `hipFileBatchIOSubmit` submits a batch of read or write operations, and
  `hipFileBatchIOGetStatus` (with a minimum of 1) waits for completions. The example refills the
  batch by submitting one replacement request for every returned event, keeping up to
  `BATCH_SIZE` requests in flight at all times.
- `BatchCookie` objects (chunk index, expected byte count) travel with each request via the
  operation's cookie pointer and let the completion path check that `status` is
  `hipFileComplete` and that `ret` matches the expected size, detecting short transfers.
- All reads complete before any write is submitted. The GPU buffer holds the whole file rounded
  up to a 4 KiB boundary (`BLOCK_ALIGN` via `align_up`), and every request is 4 KiB in size, so
  sizes stay aligned regardless of `FILE_SIZE`. The final padded write is truncated to
  `FILE_SIZE` with `ftruncate`.
- `verify_files_match` (from `examples_common`) reads both files via buffered POSIX I/O and
  compares their FNV-1a hashes.
- Both file paths must be on an `O_DIRECT`-capable local filesystem (ext4 mounted `data=ordered`,
  or xfs), and `WRITE_FILE` must differ from `READ_FILE`.

## Demonstrated API Calls

### hipFile runtime

- `hipFileBufRegister`
- `hipFileBufDeregister`
- `hipFileBatchIOSetUp`
- `hipFileBatchIOSubmit`
- `hipFileBatchIOGetStatus`
- `hipFileBatchIODestroy`

### HIP runtime

- `hipSetDevice`
- `hipMalloc`
- `hipMemset`
- `hipDeviceSynchronize`
- `hipFree`
