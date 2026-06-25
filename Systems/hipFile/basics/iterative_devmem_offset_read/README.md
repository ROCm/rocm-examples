# hipFile Iterative Device-Memory Offset Read Example

## Description

This program demonstrates reading a file into a registered GPU buffer in fixed-size chunks by
advancing the `buf_offset` parameter on each call while keeping the base device pointer fixed.
This is the registered-buffer counterpart to the `iterative_read` example, which advances the
raw pointer instead.

Using a fixed registered base pointer and advancing `buf_offset` is the preferred pattern when
the buffer has been registered with `hipFileBufRegister`, because the driver can perform
validation checks once at registration time rather than on every call.

### Application flow

1. The GPU device is selected.
2. The input file is stat'd to determine its size and the filesystem block size.
3. The input file is opened with `O_DIRECT` and registered with hipFile.
4. A device buffer is allocated with `hipMalloc`, zeroed, and registered with
   `hipFileBufRegister`.
5. A loop calls `hipFileRead` with the same base pointer (`devbuf`) on every iteration. Both
   `file_offset` and `buf_offset` advance by the number of bytes read each iteration, so each
   chunk lands at the correct offset within the registered buffer.
6. The output file is opened with `O_DIRECT` and registered. `hipFileWrite` writes the entire
   registered GPU buffer in one call.
7. `ftruncate` trims the output file to the logical payload size.
8. Both files are hashed and compared to verify correctness.
9. All resources are released in reverse order of acquisition.

## Key APIs and Concepts

- `hipFileBufRegister` registers a GPU buffer for GPU-direct I/O. With a registered buffer,
  `hipFileRead` uses the `buf_offset` parameter to address within the buffer without needing to
  modify the base pointer.
- `buf_offset` in `hipFileRead` / `hipFileWrite` specifies the byte offset within the GPU buffer
  at which the transfer begins or ends. Advancing this parameter on each iteration is more
  efficient for registered buffers than advancing the raw pointer, because the driver can cache
  the registration metadata.
- Compare with `iterative_read`, which advances the raw pointer without registration.

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
- `hipMemset`
