# hipFile AISCP Example

## Description

This example implements a minimal file-copy utility called **aiscp** (AMD Infinity Storage Copy).
It copies a source file to a destination file by routing data through GPU memory using hipFile,
demonstrating GPU-direct storage I/O in the most straightforward possible context.

For files larger than the chunk size (default: 2 GiB − 4 KiB, matching the Linux `MAX_RW_COUNT`),
the copy proceeds in chunks until the entire source has been transferred.

### Application flow

1. The source file is stat'd to determine its size and the filesystem block size.
2. The destination file is opened with `O_DIRECT` and registered with hipFile.
3. If the source is non-empty, it is opened with `O_DIRECT` and registered with hipFile.
4. A GPU buffer is allocated with `hipMalloc`, sized to `min(file_size, AISCP_CHUNK_SIZE)` rounded
   up to the filesystem block size.
5. The source is read chunk-by-chunk with `hipFileRead` into the GPU buffer. Each chunk is written
   in full to the destination with `hipFileWrite` before the next read begins.
6. After all chunks are transferred, `ftruncate` trims the destination to the exact source size
   (the final write might have been padded to satisfy the block-alignment requirement of O_DIRECT).
7. All resources are released in reverse order of acquisition.

## Key APIs and Concepts

- `hipFileHandleRegister` registers an open file descriptor with hipFile, returning a
  `hipFileHandle_t` used for all subsequent I/O on that file. The descriptor must have been opened
  with `O_DIRECT` for hipFile's GPU-direct fast path to engage.
- `hipFileHandleDeregister` releases the hipFile handle. The underlying file descriptor must still
  be closed separately with `close(2)`.
- `hipFileRead` reads up to `size` bytes from the file at `file_offset` into GPU memory at
  `buf_offset` bytes past the base of `buf`. Returns the number of bytes read, or a negative value
  on error.
- `hipFileWrite` writes up to `size` bytes to the file at `file_offset` from GPU memory at
  `buf_offset` bytes past the base of `buf`. Returns the number of bytes written, or a negative
  value on error.
- `IS_HIPFILE_ERR` / `HIPFILE_ERRSTR` are convenience macros defined in `<hipfile.h>` for
  distinguishing hipFile-internal errors from POSIX `errno`-based errors in the negative return
  value from `hipFileRead` / `hipFileWrite`.
- `hipMalloc` / `hipFree` allocate and free device (GPU) memory.

## Demonstrated API Calls

### hipFile runtime

- `hipFileHandleRegister`
- `hipFileHandleDeregister`
- `hipFileRead`
- `hipFileWrite`

### HIP runtime

- `hipMalloc`
- `hipFree`
