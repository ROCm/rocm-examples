# HIP-Doc IPC Memory Pool Shareable Handle Example

## Description

This example demonstrates how to share a memory pool between two processes using
the shareable handle IPC mechanism. The exporter creates an IPC-capable pool and
exports its handle, which the importer uses to independently allocate GPU memory
from the same pool.

Unlike the device pointer approach, a shareable handle transfers ownership of the
pool itself rather than a specific allocation. Because the handle is a POSIX file
descriptor, it must be transferred between processes using `SCM_RIGHTS` over a
Unix domain socket — writing the file descriptor integer directly to a pipe does
not work, as file descriptors are local to the process that opened them.

For more information, see
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html#shareable-handle).

### Prerequisites

The IPC memory pool API requires the `amdgpu-dkms` driver and is available only
on Linux. This example does not run on Windows.

The example consists of two cooperating processes. Run them in two terminals,
starting the exporter first:

```bash
# Terminal 1
./hip_ipc_memory_pool_shareable_handle export

# Terminal 2
./hip_ipc_memory_pool_shareable_handle import
```

### Application flow

**Exporter:**

1. A memory pool is created with `hipMemHandleTypePosixFileDescriptor` to enable IPC.
2. The pool handle is exported with `hipMemPoolExportToShareableHandle`, producing a POSIX file descriptor.
3. The file descriptor is transferred to the importer via `SCM_RIGHTS` over a Unix domain socket.
4. The exporter waits for the importer to signal completion before destroying the pool.
5. The pool and file descriptor are closed.

**Importer:**

1. The file descriptor is received via `SCM_RIGHTS` from the Unix domain socket.
2. The pool is imported with `hipMemPoolImportFromShareableHandle`.
3. A HIP stream is created.
4. Memory is allocated from the imported pool using `hipMallocFromPoolAsync`.
5. A kernel fills the allocation with computed values.
6. The stream is synchronized and results are copied to the host and verified.
7. The importer signals the exporter that it has finished.
8. Memory, pool, stream, and file descriptor are closed.

## Key APIs and Concepts

* `hipMemPoolCreate` creates a memory pool. Setting `handleTypes` to `hipMemHandleTypePosixFileDescriptor` makes the pool IPC-capable.
* `hipMemPoolExportToShareableHandle` exports the pool as a POSIX file descriptor. The descriptor must be transferred between processes via `SCM_RIGHTS` rather than by copying the integer value.
* `hipMemPoolImportFromShareableHandle` imports a pool from a received file descriptor, giving the importing process the ability to allocate GPU memory from the same pool.
* `hipMallocFromPoolAsync` allocates memory from a specific pool with stream-ordered semantics.
* `hipFreeAsync` returns memory to the pool with stream-ordered semantics.
* `hipMemPoolDestroy` destroys a memory pool.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `threadIdx`, `blockIdx`, `blockDim`

#### Host symbols

* `hipFreeAsync`
* `hipMallocFromPoolAsync`
* `hipMemPoolCreate`
* `hipMemPoolDestroy`
* `hipMemPoolExportToShareableHandle`
* `hipMemPoolImportFromShareableHandle`
* `hipMemcpy`
* `hipStreamCreate`
* `hipStreamDestroy`
* `hipStreamSynchronize`
