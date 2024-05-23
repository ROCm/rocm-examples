# HIP-Basic Streams Example

## Description

A stream encapsulates a queue of tasks that are launched on the GPU device. This example showcases usage of multiple streams, each with their own tasks. These tasks include asynchronous memory copies using `hipMemcpyAsync` and asynchronous kernel launches using `myKernelName<<<...>>>`.

### Application flow

1. Host side input and output memory is allocated using `hipHostMalloc` as pinned memory. It will ensure that the memory copies will be performed asynchronously when using `hipMemcpyAsync`.
2. Host input is instantiated.
3. Device side storage is allocated using `hipMalloc`.
4. Two `hipStream_t` streams are created using `hipStreamCreate`. The example demonstrates launching two different kernels therefore each stream queues tasks related to each kernel launch.
5. Data is copied from host to device using `hipMemcpyAsync`.
6. Two kernels, `matrix_transpose_static_shared` and `matrix_transpose_dynamic_shared` are asynchronously launched using both the streams, repectively.
7. An asynchronous memory copy task (using `hipMemcpyAsync`) is queued into the streams that transfers the results from device to host.
8. The streams are destroyed using `hipStreamDestroy`.
9. The host explicitly waits for all tasks to finish using `hipDeviceSynchronize`.
10. Free any device side memory using `hipFree`.
11. Free host side pinned memory using `hipHostFree`.

## Key APIs and Concepts

A HIP stream allows device tasks to be grouped and launched asynchronously and independently from other tasks, which can be used to hide latencies and increase task completion throughput. When results of a task queued on a particular stream are needed, it can be explicitly synchronized without blocking work queued on other streams. Each HIP stream is tied to a particular device, which enables HIP streams to be used to schedule work across multiple devices simultaneously.

## Demonstrated API Calls

### HIP runtime

- `__shared__`
- `__syncthreads`
- `hipStream_t`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipMalloc`
- `hipHostMalloc`
- `hipMemcpyAsync`
- `hipDeviceSynchronize`
- `hipFree`
- `hipHostFree`
