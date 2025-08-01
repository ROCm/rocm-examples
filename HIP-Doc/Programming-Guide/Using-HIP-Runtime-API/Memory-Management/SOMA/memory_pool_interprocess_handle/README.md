# HIP-Doc Memory Pool Interprocess Handle Example

## Description

Interprocess capable (IPC) memory pools facilitate efficient and secure sharing of GPU memory between processes.

To achieve interprocess memory sharing, you can use either [device pointers](../memory_pool_interprocess_pointer/) or
shareable handles. Both provide allocator (export) and consumer (import) interfaces.

This example demonstrates how to share handles between two processes.

### Prerequisites

This example is supported only on Linux. Make sure to launch the importer process *after* the exporter process to give
the latter some time for the setup.

### Application flow

#### Exporter

1. A memory pool is created.
2. Memory is allocated from the pool.
3. A handle is created for the allocated memory.
4. A named pipe (FIFO) is created using a Linux system call.
5. The pipe is opened. Note that the process will block until the importer has opened the pipe, too.
6. The handle data is written to the pipe.
7. The device memory is freed.
8. The memory pool is destroyed.

#### Importer

1. The pipe is opened. Note that the process will block until the exporter has opened the pipe, too.
2. Data is read from the pipe.
3. A memory pool is created.
4. The data is imported by the memory pool.
5. Device memory is allocated from the pool - the resulting pointer points to device memory allocated by the exporter.
6. The device memory is freed.
7. The memory pool is destroyed.

## Key APIs and concepts

* `hipMemPoolCreate` creates a memory pool.
* `hipMallocFromPoolAsync` is a SOMA API call that allocates device memory from a memory pool.
* `hipMemPoolExportToShareableHandle` creates an export handle for a given device pointer.
* `hipMemPoolImportFromShareableHandle` imports a handle from another process to a given memory pool.
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
* `hipMemPoolExportToShareableHandle`
* `hipMemPoolImportFromShareableHandle`
