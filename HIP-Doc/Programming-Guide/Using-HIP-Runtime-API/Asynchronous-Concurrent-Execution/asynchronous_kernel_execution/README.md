# HIP-Doc Asynchronous Kernel Execution Example

## Description

Asynchronous concurrent execution is important for efficient parallelism and resource utilization, with techniques such
as overlapping computation and data transfer, managing concurrent kernel execution with streams on single or multiple
devices, or using HIP graphs.

This example demonstrates how to execute kernels asynchronously by utilizing HIP streams. It should be compared to the
[sequential kernel execution example](../sequential_kernel_execution) and the
[event-based synchronization example](../event_based_synchronization).

### Application flow

1. Two data vectors are created both on the host and on the device.
2. Two streams are created.
3. Loop for a fixed number of iterations:
   1. Copy the vector contents from the host to the device. The first vector is copied in the first stream and the
      second vector in the second stream.
   2. Launch the first GPU kernel on the first stream which operates on the first vector and writes the result back to
      the same vector.
   3. Launch the second GPU kernel on the second stream which operates on both vectors and writes the result back to
      the second vector.
   4. Copy the results back to the corresponding host vectors. The first vector is copied in the first stream and the
      second vector in the second stream.
4. The streams and the host are synchronized.
5. The results are verified.
6. The streams are destroyed and the device memory is freed.

## Key APIs and Concepts

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

* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipLaunchKernelGGL`
* `hipMemcpyAsync`
* `hipStreamCreate`
* `hipStreamDestroy`
* `hipStreamSynchronize`
