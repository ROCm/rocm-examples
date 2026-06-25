# hipFile Various Memory Types Read/Write Example

## Description

This program demonstrates that hipFile can transfer data through three different memory backing
types: device memory (allocated with `hipMalloc`), managed memory (`hipMallocManaged`), and
pinned host memory (`hipHostMalloc`). A command-line `MODE` argument selects the memory type at
runtime. The transfer path (file → GPU buffer → file) and the verification step are identical
across all three modes.

### Application flow

1. The GPU device is selected.
2. The input file is stat'd to determine its size and the filesystem block size.
3. The input file is opened with `O_DIRECT` and registered with hipFile.
4. A buffer of the type selected by `MODE` is allocated and zeroed.
5. `hipFileRead` reads the input file into the buffer in a single call.
6. The output file is opened with `O_DIRECT` and registered. `hipFileWrite` writes the buffer to
   the output file in a single call. `ftruncate` trims the file to the logical payload size.
7. Both files are hashed via plain POSIX I/O. The hashes are compared to verify correctness.
8. All resources are released using the appropriate free function for the selected memory type.

## Key APIs and Concepts

- **Device memory** (`hipMalloc`): The standard GPU allocation. hipFile uses GPU-direct I/O when
  reading into or writing from device memory. Explicit buffer registration is not required.
- **Managed memory** (`hipMallocManaged`): Unified memory accessible from both CPU and GPU.
  hipFile transfers managed memory without requiring explicit buffer registration. Page migration
  may occur depending on hardware and driver support.
- **Pinned host memory** (`hipHostMalloc`): Memory pinned in host physical RAM and accessible by
  the GPU via DMA. hipFile can use it for transfers without explicit buffer registration, though
  performance characteristics differ from device memory.
- hipFile accepts pointers from all three allocators without explicit `hipFileBufRegister`,
  making it straightforward to experiment with different memory placement strategies.

## Demonstrated API Calls

### hipFile runtime

- `hipFileHandleRegister`
- `hipFileHandleDeregister`
- `hipFileRead`
- `hipFileWrite`

### HIP runtime

- `hipSetDevice`
- `hipMalloc` / `hipFree`
- `hipMallocManaged` / `hipFree`
- `hipHostMalloc` / `hipHostFree`
- `hipMemset`
- `hipMemAttachGlobal`
- `hipHostMallocDefault`
