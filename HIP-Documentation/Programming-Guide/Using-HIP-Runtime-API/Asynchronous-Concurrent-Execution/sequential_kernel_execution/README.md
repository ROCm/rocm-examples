# HIP-Documentation Sequential Kernel Execution Example

## Description

Asynchronous concurrent execution is important for efficient parallelism and resource utilization, with techniques such
as overlapping computation and data transfer, managing concurrent kernel execution with streams on single or multiple
devices, or using HIP graphs.

This example demonstrates how to execute kernels sequentially, i.e. without any overlap with other kernels or device
operations. It should be compared to the [asynchronous kernel execution example](../asynchronous_kernel_execution) and
the [event-based synchronization example](../event_based_synchronization).

### Application flow

1. Two data vectors are created both on the host and on the device.
2. Loop for a fixed number of iterations:
    1. Copy the vector contents from the host to the device.
    2. Launch the first GPU kernel which operates on the first vector and writes the result back to the same vector.
    3. Launch the second GPU kernel which operates on both vectors and writes the result back to the second vector.
    4. Copy the results back to the corresponding host vectors.
3. Device and host are synchronized.
4. The results are verified.
5. The device memory is freed.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `threadIdx`

#### Host symbols

* `hipDeviceSynchronize`
* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipLaunchKernelGGL`
* `hipMemcpy`
