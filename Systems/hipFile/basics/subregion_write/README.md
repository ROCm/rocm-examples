# hipFile Subregion Write Example

## Description

This example demonstrates using hipFile's `buf_offset` parameter in `hipFileWrite` to write only
a trailing sub-region of a GPU buffer to an output file. The full input file is first read into
a registered GPU buffer, and then only the bytes at or after `SW_SUB_OFFSET` (default 8192 bytes)
are written to the output, using `buf_offset` to skip the leading portion without moving data
in the GPU buffer.

### Application flow

1. The GPU device is selected.
2. The input file is stat'd to determine its size and the filesystem block size.
3. The input file is opened with `O_DIRECT` and registered with hipFile.
4. A device buffer is allocated, registered with `hipFileBufRegister`, and zeroed.
5. `hipFileRead` reads the entire input file into the GPU buffer in a single call.
6. The output file is opened with `O_DIRECT` and registered.
7. `hipFileWrite` is called with `buf_offset = SW_SUB_OFFSET` and `file_offset = 0`. This causes
   the driver to read from `devbuf[SW_SUB_OFFSET .. SW_SUB_OFFSET + write_xfer)` and write it
   to the output file starting at byte 0.
8. `ftruncate` trims the output file to the exact sub-region size (`payload_size - SW_SUB_OFFSET`).
9. The sub-region of the input file (bytes `[SW_SUB_OFFSET, payload_size)`) and the entire output
   file are hashed and compared to verify correctness.
10. All resources are released.

## Key APIs and Concepts

- `buf_offset` in `hipFileWrite` specifies the byte offset within the registered GPU buffer from
  which the transfer begins. Combined with a zero `file_offset`, this lets the caller write a
  contiguous sub-region of the GPU buffer to the beginning of the output file without any
  intermediate data movement.
- Because the write is issued with `O_DIRECT`, `SW_SUB_OFFSET` must be a multiple of the
  filesystem's logical block size.
- `hipFileBufRegister` is required for non-zero `buf_offset` to work correctly on the GPU-direct
  fastpath backend.

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
