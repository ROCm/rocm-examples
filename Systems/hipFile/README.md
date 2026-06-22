# hipFile Examples

This directory contains working examples of the hipFile API, grouped by what they demonstrate.
Every program verifies its result with an FNV-1a hash and prints `OK …` on success.

Most examples move data through the GPU on hipFile's fast path, which opens files with `O_DIRECT`.
Running them therefore requires an AMD GPU supported by ROCm and source/destination paths on an
`O_DIRECT`-capable local filesystem (ext4 mounted `data=ordered`, or xfs). `O_DIRECT` is not a
hipFile requirement — files can be opened without it and routed through the POSIX compat path (see
[`basics/no_odirect_write`](basics/no_odirect_write)). Verify fast-path support with
`/opt/rocm/bin/ais-check`. The [`api`](api) examples are the exception: they only query the
library and require neither a GPU nor an `O_DIRECT` filesystem.

## Prerequisites

- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 7.x)
- hipFile (installed via the nightly `.deb` packages or built from source)
- CMake (at least version 3.21), or GNU Make

## Directory layout

| Directory | What's in it |
| --------- | ------------ |
| [`api`](api) | Minimal examples of the non-I/O API — calls that query or configure the library (e.g. `get_version`). No `O_DIRECT` filesystem or file arguments needed. |
| [`basics`](basics) | Small, single-purpose programs that each exercise one facet of the synchronous API: buffer registration, the `O_DIRECT` fast path vs. compat fallback, chunked reads, device-buffer offsets, sub-region writes, GPU memory types, and a full round trip. |
| [`async`](async) | Examples of the asynchronous, stream-based API (`hipFileReadAsync` / `hipFileWriteAsync`), including single-stream, non-blocking-stream, and concurrent multi-stream round trips. |
| [`aiscp`](aiscp) | A standalone `cp`-like utility built on hipFile (`hipfile_aiscp SOURCE DEST`). |
| [`common`](common) | Shared helpers used by `basics` and `async` (alignment math, pattern fill, FNV-1a hashing, file open/register). Not an example — compiled directly into each example that needs it. |

## Building

### CMake (from the repository root)

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --parallel
```

The binaries land under `build/bin/Systems/hipFile/`.

### CMake (Systems subtree only)

```bash
cd Systems
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --parallel
```

### CMake (single example)

Each example directory is a self-contained CMake project. For example:

```bash
cd Systems/hipFile/basics/bufregister_write
cmake -S . -B build
cmake --build build
```

### Make

```bash
cd Systems
make -j $(nproc)
```

Or for a single category:

```bash
cd Systems/hipFile/basics
make -j $(nproc)
```

---

## `api`

Minimal examples of the non-I/O parts of the hipFile API — calls that query or configure the
library rather than move data through the GPU. These do **not** require an `O_DIRECT`-capable
filesystem or file arguments.

| Program | What it shows | Args |
| ------- | ------------- | ---- |
| [`get_version`](api/get_version) | Read the hipFile version both ways: the `HIPFILE_VERSION_*` header macros (compile-time) and `hipFileGetVersion()` (runtime). | none |

### Running

```bash
./hipfile_get_version
```

Prints the version from the header symbols and from the runtime call. No file or GPU memory is
touched.

---

## `basics`

Small, single-purpose programs that each exercise one facet of the synchronous hipFile C API.
They drive the API directly from `main()` and use the shared helpers in
[`common`](common). Every example verifies its result with an FNV-1a hash and prints `OK …` on
success.

| Program | What it shows | Args |
| ------- | ------------- | ---- |
| [`bufregister_write`](basics/bufregister_write) | Write a GPU buffer registered with `hipFileBufRegister` straight to disk (the fast path). | `OUTPUT [GPUID]` |
| [`no_bufregister_write`](basics/no_bufregister_write) | Same write, but without registering the buffer — hipFile copies through its internal pool. | `OUTPUT [GPUID]` |
| [`no_odirect_write`](basics/no_odirect_write) | Register a file opened *without* `O_DIRECT`. hipFile transparently reopens with `O_DIRECT` on capable filesystems; falls back to the POSIX compat path otherwise. | `OUTPUT [GPUID]` |
| [`iterative_read`](basics/iterative_read) | Chunked read into GPU memory where the **host pointer** advances each iteration, then one write. | `INPUT OUTPUT [GPUID]` |
| [`iterative_devmem_offset_read`](basics/iterative_devmem_offset_read) | Same chunked read, but the base device pointer is fixed and the **`buf_offset`** argument advances. | `INPUT OUTPUT [GPUID]` |
| [`subregion_write`](basics/subregion_write) | Read a whole file, then write only the bytes at/after an offset using a non-zero `buffer_offset`. | `INPUT OUTPUT [GPUID]` |
| [`various_mem_rw`](basics/various_mem_rw) | Round-trip a file using device (`1`), managed (`2`), or pinned-host (`3`) memory as the transfer buffer. | `INPUT OUTPUT MODE [GPUID]` |
| [`roundtrip_verify`](basics/roundtrip_verify) | Write a known pattern, read it back through the GPU, write a copy, and assert both files hash-match. | `CREATED COPIED [GPUID]` |

`GPUID` is optional and defaults to `0`. Payload and chunk sizes are compile-time `#define`s
(e.g. `-DBRW_SIZE=…`, `-DIR_CHUNK_SIZE=…`) documented at the top of each `.cpp`.

### Running

Examples that read an existing input file (`iterative_read`, `iterative_devmem_offset_read`,
`subregion_write`, `various_mem_rw`) need the input to exist first. Create one with `dd`:

```bash
dd if=/dev/urandom of=input.bin bs=1M count=1
```

The input and output paths must live on an `O_DIRECT`-capable local filesystem (ext4 mounted
`data=ordered`, or xfs). Verify with `/opt/rocm/bin/ais-check`. Then:

```bash
./hipfile_bufregister_write            out_bufregister.bin
./hipfile_no_bufregister_write         out_no_bufregister.bin
HIPFILE_ALLOW_COMPAT_MODE=1 \
  ./hipfile_no_odirect_write           out_no_odirect.bin
./hipfile_iterative_read               input.bin out_iter.bin
./hipfile_iterative_devmem_offset_read input.bin out_iter_off.bin
./hipfile_subregion_write              input.bin out_subregion.bin
./hipfile_various_mem_rw               input.bin out_vmrw.bin 1   # 1=device 2=managed 3=pinned
./hipfile_roundtrip_verify             rtv_created.bin rtv_copied.bin
```

---

## `async`

Examples of hipFile's asynchronous, stream-based API (`hipFileReadAsync` /
`hipFileWriteAsync`). Each example seeds an input file with a deterministic pattern, issues a
GPU-mediated read+write round trip on one or more HIP streams, synchronizes, and verifies the
output by FNV-1a hash. They share the helpers in [`common`](common) and print `OK …` on success.

> **Note:** The `O_DIRECT` fast path is not currently supported for asynchronous I/O — async
> operations always run through the POSIX compat (fallback) path, regardless of whether the file
> or filesystem is `O_DIRECT`-capable. Fast-path async support is planned for the future.

| Program | What it shows |
| ------- | ------------- |
| [`roundtrip_async`](async/roundtrip_async) | Async read + write on the **default stream**, a single `hipStreamSynchronize`, then verify. |
| [`roundtrip_async_nonblocking_stream`](async/roundtrip_async_nonblocking_stream) | Same round trip on an explicit `hipStreamNonBlocking` stream (no implicit sync with the legacy default stream). |
| [`roundtrip_async_multi_stream`](async/roundtrip_async_multi_stream) | `NUM_STREAMS` read/write pairs run concurrently, each on its own non-blocking stream covering a distinct file slice. |
| [`roundtrip_async_multi_stream_registered`](async/roundtrip_async_multi_stream_registered) | Same concurrent multi-stream run, but each stream is registered with `hipFileStreamRegister` (fixed-offset / fixed-size / page-aligned hints) so the driver skips per-submission validation. |

All four take the same arguments:

```text
PROGRAM READ_FILE WRITE_FILE [GPUID]
```

`READ_FILE` is created and seeded by the program itself, so it does **not** need to exist
beforehand. `WRITE_FILE` receives the round-tripped payload. `GPUID` is optional (default `0`).
Sizes and stream counts are compile-time `#define`s documented at the top of each `.cpp`.

### Running

Both file paths must live on a local filesystem. From the directory containing the built
binaries:

```bash
./hipfile_roundtrip_async                         in.bin out.bin
./hipfile_roundtrip_async_nonblocking_stream      in.bin out.bin
./hipfile_roundtrip_async_multi_stream            in.bin out.bin
./hipfile_roundtrip_async_multi_stream_registered in.bin out.bin
```

---

## `aiscp`

`hipfile_aiscp` is a simple file-copy utility built on hipFile. It routes data through GPU memory
and works like the Linux `cp` command:

```bash
hipfile_aiscp SOURCE DEST
```

SOURCE must be an existing file. DEST is created or overwritten. Both paths must live on an
`O_DIRECT`-capable local filesystem.

---

## `common`

Not an example — a small collection of shared helpers used by `basics` and `async`:

- `align_up` / `is_power_of_two` — alignment utilities
- `fill_pattern` — fills a buffer with a deterministic byte pattern
- `hash_buffer` / `hash_file_range` — FNV-1a hashing for verification
- `seed_read_file` — creates a test input file via plain POSIX I/O
- `verify_files_match` — hashes two files and asserts they are equal
- `open_file` / `close_file` — wraps `open(2)` + `hipFileHandleRegister` / deregister

See [`common/examples_common.h`](common/examples_common.h) for the full documented API.
