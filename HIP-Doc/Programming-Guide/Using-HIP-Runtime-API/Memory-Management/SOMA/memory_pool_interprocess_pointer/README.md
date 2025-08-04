# HIP-Doc Memory Pool Interprocess Pointer Example

## Description

This example demonstrates how to share device pointers between two processes. For more information on this topic, please
refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html#interprocess-memory-handling).

### Prerequisites

This example is supported only on Linux. Make sure to launch the importer process *after* the exporter process to give
the latter some time for the setup.

### Application flow

#### Exporter

1. Device memory is allocated.
2. Export data is generated for the resulting pointer.
3. A named pipe (FIFO) is created using a Linux system call.
4. The pipe is opened. Note that the process will block until the importer has opened the pipe, too.
5. The export data is written to the pipe.
6. The device memory is freed.

#### Importer

1. The pipe is opened. Note that the process will block until the exporter has opened the pipe, too.
2. Data is read from the pipe.
3. A memory pool is created.
4. The data is imported by the memory pool; the result is a pointer to device memory allocated by the exporter.
5. The device memory is freed.
6. The memory pool is destroyed.

## Key APIs and concepts

* `hipMalloc` allocates device memory.
* `hipMemPoolCreate` creates a memory pool.
* `hipMemPoolExportPointer` creates export data for a given device pointer.
* `hipMemPoolImportPointer` imports a pointer from another process to a given memory pool.
* `hipFree` frees device memory and (in this case) returns it to a memory pool.
* `hipMemPoolDestroy` destroys a memory pool.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipMemPoolCreate`
* `hipMemPoolDestroy`
* `hipMemPoolExportPointer`
* `hipMemPoolImportPointer`
