# HIP-Basic Events Example

## Description

Memory transfer and kernel execution are the most important parameters in parallel computing, especially in high performance computing (HPC) and machine learning. Memory bottlenecks are the main problem why we are not able to get the highest performance, therefore obtaining the memory transfer timing and kernel execution timing plays key role in application optimization.

This example showcases measuring kernel and memory transfer timing using HIP events. The kernel under measurement is a trivial one that performs square matrix transposition.

### Application flow

1. A number of parameters are defined that control the problem details and the kernel launch.
2. Input data is set up in host memory.
3. The necessary amount of device memory is allocated.
4. A pair of `hipEvent` objects are defined and initialized.
5. Time measurement is started on the `start` event.
6. Memory transfer from host to device of the input data is performed.
7. The time measurement is stopped using the `stop` event. The execution time is calculated via the `start` and `stop` events and it is printed to the standard output.
8. The kernel is launched, and its runtime is measured similarly using the `start` and `stop` events.
9. The result data is copied back to the host, and the execution time of the copy is measured similarly.
10. The allocated device memory is freed and the event objects are released.
11. The result data is validated by comparing it to the product of the reference (host) implementation. The result of the validation is printed to the standard output.

## Key APIs and Concepts

- The `hipEvent_t` type defines HIP events that can be used for synchronization and time measurement. The events must be initialized using `hipEventCreate` before usage and destroyed using `hipEventDestroy` after they are no longer needed.
- The events have to be queued on a device stream in order to be useful, this is done via the `hipEventRecord` function. The stream itself is a list of jobs (memory transfers, kernel executions and events) that execute sequentially. When the event is processed by the stream, the current machine time is recorded to the event. This can be used to measure execution times on the stream. In this example, the default stream is used.
- The time difference between two recorded events can be accessed using the function `hipEventElapsedTime`.
- An event can be used to synchronize the execution of the jobs on a stream with the execution of the host. A call to `hipEventSynchronize` blocks the host until the provided event is scheduled on its stream.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`

#### Host symbols

- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipEventCreate`
- `hipEventRecord`
- `hipEventElapsedTime`
- `hipEventSynchronize`
