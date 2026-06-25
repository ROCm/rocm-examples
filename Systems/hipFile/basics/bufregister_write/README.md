# hipFile Bufregister Write Example

## Description

This program demonstrates writing a GPU memory buffer to a file using hipFile with explicit buffer
registration. Registering the GPU buffer with `hipFileBufRegister` before calling `hipFileWrite`
enables hipFile to take the GPU-direct I/O fast path, bypassing the internal pool copy that would
otherwise be required for unregistered buffers.

### Application flow

1. The GPU device is selected.
2. A deterministic byte pattern (default 128 KiB) is generated in CPU memory and copied to a
   GPU buffer allocated with `hipMalloc`.
3. The GPU buffer is registered with `hipFileBufRegister`.
4. The output file is opened with `O_DIRECT` and registered with hipFile via
   `hipFileHandleRegister`.
5. `hipFileWrite` writes the block-aligned buffer to the file.
6. `ftruncate` trims the file to the exact logical payload size (the write may have been rounded up
   to the block boundary required by `O_DIRECT`).
7. The CPU pattern is hashed and the written file is read back and hashed via plain POSIX I/O.
   The hashes are compared to verify the transfer was lossless.
8. All resources are released: file handle, buffer registration, device buffer, CPU buffer.

## Key APIs and Concepts

- `hipFileBufRegister` pins a GPU memory region so that hipFile can use it for GPU-direct
  transfers without going through an intermediate bounce buffer. The region must stay valid until
  `hipFileBufDeregister` is called.
- `hipFileBufDeregister` releases the registration. The underlying `hipMalloc` allocation is
  still valid and must be freed separately with `hipFree`.
- `hipFileHandleRegister` registers an `O_DIRECT` file descriptor with hipFile, enabling
  GPU-direct I/O on that file. The descriptor must be opened with `O_DIRECT` for the fast path.
- `hipFileWrite` performs a synchronous GPU-to-file transfer from the registered GPU buffer.
  The transfer size must be a multiple of the filesystem's logical block size when `O_DIRECT` is
  in use; use `ftruncate` after writing to set the exact file size.
- `IS_HIPFILE_ERR` / `HIPFILE_ERRSTR` distinguish hipFile-internal errors from POSIX `errno`
  errors in the negative return values of `hipFileRead` and `hipFileWrite`.

## Demonstrated API Calls

### hipFile runtime

- `hipFileBufRegister`
- `hipFileBufDeregister`
- `hipFileHandleRegister`
- `hipFileHandleDeregister`
- `hipFileWrite`

### HIP runtime

- `hipSetDevice`
- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
