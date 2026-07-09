# HIP-Doc IPC Memory Pool Device Pointer Example

## Description

This example demonstrates how to share a stream ordered memory allocation between
two processes using the device pointer IPC mechanism. The exporter allocates memory
from an IPC-capable pool, fills it using a GPU kernel, and exports an opaque handle
(`hipMemPoolPtrExportData`) that the importer uses to access the same allocation.

For more information, see
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html#device-pointer).

### Prerequisites

The IPC memory pool API requires the `amdgpu-dkms` driver and is available only
on Linux. This example does not run on Windows.

The example consists of two cooperating processes. Run them in two terminals,
starting the exporter first:

```bash
# Terminal 1
./hip_ipc_memory_pool_device_pointer export

# Terminal 2
./hip_ipc_memory_pool_device_pointer import
```

### Application flow

**Exporter:**

1. A memory pool is created with `hipMemHandleTypePosixFileDescriptor` to enable IPC.
2. A HIP stream is created.
3. Memory is allocated from the pool using `hipMallocFromPoolAsync`.
4. A kernel fills the allocation with computed values.
5. The stream is synchronized.
6. The device pointer is exported with `hipMemPoolExportPointer`, producing a serializable `hipMemPoolPtrExportData` struct.
7. The struct is written to a named pipe for the importer to read.
8. The exporter waits for the importer to signal completion before freeing memory.
9. Memory, pool, and stream are destroyed.

**Importer:**

1. The `hipMemPoolPtrExportData` struct is read from the named pipe.
2. A memory pool is created with matching IPC-capable properties.
3. The device pointer is imported with `hipMemPoolImportPointer`.
4. The imported pointer is copied to host memory and verified.
5. The importer signals the exporter that it has finished.
6. The imported pointer and pool are destroyed.

## Key APIs and Concepts

* `hipMemPoolCreate` creates a memory pool. Setting `handleTypes` to `hipMemHandleTypePosixFileDescriptor` makes the pool IPC-capable.
* `hipMallocFromPoolAsync` allocates memory from a specific pool with stream-ordered semantics.
* `hipMemPoolExportPointer` exports an allocation as a serializable `hipMemPoolPtrExportData` struct that can be written to any IPC channel and read by another process.
* `hipMemPoolImportPointer` imports a previously exported allocation into a local pool handle, giving the importing process access to the same GPU memory.
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
* `hipMemPoolExportPointer`
* `hipMemPoolImportPointer`
* `hipMemcpy`
* `hipStreamCreate`
* `hipStreamDestroy`
* `hipStreamSynchronize`
