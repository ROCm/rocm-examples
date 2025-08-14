# HIP-Doc Sequential Kernel Execution Example

## Description

This example demonstrates how to execute HIP operations and kernels sequentially. For more
information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html#asynchronous-concurrent-execution).

This example should be compared to the [asynchronous kernel execution example](../async_kernel_execution) and the
[event-based synchronization example](../event_based_synchronization).

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
