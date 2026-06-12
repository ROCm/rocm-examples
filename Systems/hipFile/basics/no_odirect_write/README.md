# hipFile No-O_DIRECT Write Example

## Description

This program demonstrates what happens when hipFile is used with a file opened *without*
`O_DIRECT`. Because the GPU-direct fast path requires `O_DIRECT` to bypass the page cache,
omitting it causes hipFile to fall back to a POSIX-compatible I/O path instead.

The compat path must be explicitly enabled at runtime by setting the environment variable
`HIPFILE_ALLOW_COMPAT_MODE=1`. Without it, hipFile will reject the operation.

### Application flow

1. The GPU device is selected.
2. A deterministic byte pattern (default 128 KiB) is generated in CPU memory and copied to a
   GPU buffer allocated with `hipMalloc`.
3. The GPU buffer is registered with `hipFileBufRegister`.
4. The output file is opened **without** `O_DIRECT` and registered with hipFile via
   `hipFileHandleRegister`.
5. `hipFileWrite` is called. Because the file descriptor lacks `O_DIRECT`, hipFile routes the
   write through a POSIX-compatible bounce path.
6. `ftruncate` trims the file to the exact logical payload size (defensive; the compat path can
   write the exact size, so the truncate is a no-op here).
7. The written file is read back via plain POSIX I/O and its hash is compared against the expected
   pattern hash.
8. All resources are released.

## Key APIs and Concepts

- `O_DIRECT` is required for the hipFile GPU-direct fast path. Files opened without it trigger
  the POSIX compatibility path, which performs an extra data copy through the kernel page cache.
- `HIPFILE_ALLOW_COMPAT_MODE=1` enables the compat path. Without it, hipFile rejects writes to
  non-`O_DIRECT` file handles to prevent accidental performance degradation.
- Unlike the `O_DIRECT` path, the compat path does not require transfer sizes to be a multiple of
  the filesystem's logical block size, so the exact payload size can be written directly.

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
