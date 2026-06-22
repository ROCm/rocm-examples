# hipFile No-O_DIRECT Write Example

## Description

This program demonstrates what happens when hipFile is used with a file opened *without*
`O_DIRECT`. On an `O_DIRECT`-capable filesystem, hipFile transparently reopens the file with
`O_DIRECT` and still takes the GPU-direct fast path. On a filesystem that cannot support
`O_DIRECT`, hipFile falls back to its POSIX-compatible I/O path instead.

The compat path is enabled by default. The environment variable `HIPFILE_ALLOW_COMPAT_MODE` only
needs to be set to `true` if you previously disabled compat mode and want to re-enable it.

### Application flow

1. The GPU device is selected.
2. A deterministic byte pattern (default 128 KiB) is generated in CPU memory and copied to a
   GPU buffer allocated with `hipMalloc`.
3. The GPU buffer is registered with `hipFileBufRegister`.
4. The output file is opened **without** `O_DIRECT` and registered with hipFile via
   `hipFileHandleRegister`.
5. `hipFileWrite` is called. On an `O_DIRECT`-capable filesystem, hipFile transparently reopens
   the file with `O_DIRECT` and takes the fast path. On a filesystem that cannot support
   `O_DIRECT`, hipFile routes the write through a POSIX-compatible bounce path.
6. `ftruncate` trims the file to the exact logical payload size (defensive; the compat path can
   write the exact size, so the truncate is a no-op here).
7. The written file is read back via plain POSIX I/O and its hash is compared against the expected
   pattern hash.
8. All resources are released.

## Key APIs and Concepts

- When a file is opened without `O_DIRECT`, hipFile first attempts to reopen it with `O_DIRECT` to
  engage the GPU-direct fast path. If the filesystem does not support `O_DIRECT`, hipFile falls
  back to the POSIX compatibility path, which performs an extra data copy through the kernel page
  cache.
- The compat path is enabled by default. Set `HIPFILE_ALLOW_COMPAT_MODE=true` only if you have
  previously disabled it and need to re-enable it.
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
