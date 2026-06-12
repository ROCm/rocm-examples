# hipFile No-Bufregister Write Example

## Description

This program demonstrates writing a GPU buffer to a file using hipFile *without* explicitly
registering the buffer with `hipFileBufRegister`. When the source buffer is unregistered, hipFile
routes the data through its own internal pool buffer, performing an extra copy from device to the
pool before the I/O is issued. The result is identical to the registered case but the transfer
goes through a bounce path.

This example is useful for understanding the behavioral difference between registered and
unregistered GPU buffers, and for testing environments where buffer registration is not available
or desired.

### Application flow

1. The GPU device is selected.
2. A deterministic byte pattern (default 1 MiB) is generated in CPU memory and copied to a
   GPU buffer allocated with `hipMalloc`.
3. The output file is opened with `O_DIRECT` and registered with hipFile via
   `hipFileHandleRegister` (only the file handle is registered — the GPU buffer is not).
4. `hipFileWrite` writes the block-aligned buffer to the file. Because the buffer is unregistered,
   hipFile copies through its internal pool buffer.
5. `ftruncate` trims the file to the exact logical payload size.
6. The written file is read back via plain POSIX I/O and its hash is compared against the expected
   pattern hash to verify correctness.
7. All resources are released.

## Key APIs and Concepts

- When `hipFileWrite` is called with an unregistered GPU buffer, hipFile does not use GPU-direct
  I/O. Instead it copies the data through an internal pool buffer. This is the compatibility path
  for buffers that cannot or have not been registered.
- Compare with the `bufregister_write` example, which uses `hipFileBufRegister` to enable the
  GPU-direct fast path and avoid the internal copy.
- `hipFileHandleRegister` registers only the file descriptor. The GPU buffer registration is
  independent and optional.

## Demonstrated API Calls

### hipFile runtime

- `hipFileHandleRegister`
- `hipFileHandleDeregister`
- `hipFileWrite`

### HIP runtime

- `hipSetDevice`
- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
