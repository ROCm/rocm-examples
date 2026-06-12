# hipFile Roundtrip Verify Example

## Description

This program performs a complete GPU-mediated read-write round trip and verifies data integrity
with hashing. It writes a known pattern to disk via hipFile, reads it back into the same GPU
buffer, writes it to a second file, and then asserts both files have identical contents. This is
the canonical correctness test for hipFile I/O.

### Application flow

1. The GPU device is selected.
2. A deterministic byte pattern (default 64 KiB) is generated in CPU memory, copied to a GPU
   buffer with `hipMalloc` and `hipMemcpy`, and the GPU buffer is registered with
   `hipFileBufRegister`.
3. **Phase 1 – Write:** The output file `CREATED` is opened with `O_DIRECT` and registered.
   `hipFileWrite` writes the block-aligned GPU buffer to the file. `ftruncate` trims it to the
   logical payload size. The file is closed.
4. **Phase 2 – Read-back:** `CREATED` is reopened for reading with `O_DIRECT` and registered.
   `hipFileRead` reads the file contents back into the *same* GPU buffer, overwriting it. If the
   round trip is lossless, the GPU buffer now holds the same bytes that were written.
5. **Phase 3 – Copy:** The output file `COPIED` is opened with `O_DIRECT` and registered.
   `hipFileWrite` writes the read-back GPU buffer to `COPIED`. `ftruncate` trims the file.
6. Both files are read via plain POSIX I/O, hashed with FNV-1a, and the hashes are compared.
   A match confirms the entire write-read-write cycle was lossless.
7. All resources are released.

## Key APIs and Concepts

- `hipFileBufRegister` / `hipFileBufDeregister` register and deregister the GPU buffer for
  GPU-direct transfers.
- `hipFileWrite` writes from GPU memory to a file. The file must be open with `O_DIRECT` for
  the GPU-direct path. Transfer sizes must be block-aligned; `ftruncate` corrects the file size.
- `hipFileRead` reads from a file into GPU memory. The same block-alignment requirements apply.
- Reusing the same GPU buffer for both the write and the subsequent read-back is intentional:
  it confirms that the on-disk contents are exactly what was written, independent of any
  in-memory state.
- The `verify_files_match` helper (from `examples_common`) reads both files via buffered POSIX
  I/O and compares their FNV-1a hashes.

## Demonstrated API Calls

### hipFile runtime

- `hipFileBufRegister`
- `hipFileBufDeregister`
- `hipFileHandleRegister`
- `hipFileHandleDeregister`
- `hipFileRead`
- `hipFileWrite`

### HIP runtime

- `hipSetDevice`
- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
