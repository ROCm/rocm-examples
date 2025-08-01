# HIP-Doc Event-Based Synchronization Example

## Description

Asynchronous concurrent execution is important for efficient parallelism and resource utilization. This includes
techniques such as overlapping computation and data transfer, managing concurrent kernel execution with streams on
single or multiple devices, or using HIP graphs.

This example demonstrates how to execute kernels asynchronously by utilizing HIP streams and events. It should be
compared to the [sequential kernel execution example](../sequential_kernel_execution) and the
[asynchronous kernel execution example](../asynchronous_kernel_execution).

### Application flow

1. Two data vectors are created, both on the host and on the device.
2. Two streams are created.
3. Three events are created.
4. A loop is executed for a fixed number of iterations:
   1. The vector contents are copied from the host to the device. The first vector is copied in the first stream, and
      the second vector is copied in the second stream.
   2. The first GPU kernel is launched on the first stream, which operates on the first vector and writes the result
      back to the same vector.
   3. An event is added to the first stream that will be reached after the kernel completes.
   4. The first stream's event is waited for in the second stream before continuing.
   5. The second GPU kernel is launched on the second stream, which operates on both vectors and writes the result back
      to the second vector.
   6. The results are copied back to the corresponding host vectors. The first vector is copied in the first stream, and
      the second vector is copied in the second stream.
   7. Events are added to both streams. These will be reached after the copy operations finish.
   8. Both events are waited for by the respective other stream before continuing.
5. The results are verified.
6. The events and streams are destroyed, and the device memory is freed.

## Key APIs and Concepts

* `hipEventCreate` creates an event. This can be placed in a stream. Other streams (or the host) can wait for this
   event to be reached, offering another form of synchronization.
* `hipEventDestroy` destroys an event.
* `hipStreamCreate` creates a stream. A stream executes HIP operations asynchronously with regard to the host.
* `hipStreamDestroy` destroys a stream.
* `hipMemcpyAsync` performs a copy operation in a stream. The call returns immediately to the host and the copy
  operation is performed asynchronously.
* `hipLaunchKernelGGL` can launch a kernel in a stream.
* `hipStreamSynchronize` blocks the host until all operations in the given stream have finished.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `threadIdx`

#### Host symbols

* `hipEventCreate`
* `hipEventDestroy`
* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipLaunchKernelGGL`
* `hipMemcpyAsync`
* `hipStreamCreate`
* `hipStreamDestroy`
* `hipStreamSynchronize`
