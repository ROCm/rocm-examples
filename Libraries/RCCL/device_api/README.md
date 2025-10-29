# RCCL Device API Concepts and Kernel Fusion

## Description

This example demonstrates RCCL device-side API concepts and the benefits of fusing computation with collective communication operations.

The Device API aims to eliminate the separation between computation and communication:

$Output_{i} = Reduce_{j=0}^{N-1} Computation(Input_{i,j})$

where

- $Input_{i,j}$ is input data for computation on rank $i$ from rank $j$
- $Computation()$ represents device-side operations (e.g., gradient clipping)
- $Reduce()$ is the collective reduction operation
- $Output_{i}$ is the final result on rank $i$
- $N$ is the number of ranks in the communicator
- Device API enables fusing $Computation()$ and $Reduce()$ into a single kernel

## Application flow

1. Set up the number of ranks, array size, and gradient clipping threshold.
2. Detect available GPUs and configure the number of ranks.
3. Initialize RCCL communicators for each GPU using `ncclCommInitAll()`.
4. Query communicator properties using `ncclCommUserRank()`, `ncclCommCount()`, and `ncclCommCuDevice()`.
5. Create HIP streams for each rank to enable asynchronous operations.
6. Allocate device memory for multiple buffers (input, clipped, output) using `ncclMemAlloc()`.
7. Initialize host data with gradient-like patterns and copy to device memory.
8. Launch the gradient clipping kernel (`gradient_clip_kernel`) to perform computation on device.
9. Perform the AllReduce operation using `ncclGroupStart()` and `ncclGroupEnd()` for optimized execution.
10. Execute `ncclAllReduce()` to reduce the clipped gradients across all ranks.
11. Synchronize all streams to ensure completion of kernel and collective operations.
12. Copy results back to host memory and verify the computation and reduction results.
13. Clean up resources in the proper order: destroy streams, free memory, finalize and destroy communicators.

## Key APIs and Concepts

- **RCCL Initialization**: The RCCL library is initialized by creating handles with `ncclCommInitAll()` for all ranks simultaneously and released with `ncclCommDestroy()` after finalization with `ncclCommFinalize()`.

- **Communicator Management**:
  - `ncclCommInitAll()`: Creates and initializes communicators for all specified ranks with automatic device assignment.
  - `ncclCommUserRank()`: Queries the user-assigned rank ID for a specific communicator.
  - `ncclCommCount()`: Returns the total number of ranks in the communicator.
  - `ncclCommCuDevice()`: Retrieves the CUDA device ID associated with a communicator.
  - `ncclCommFinalize()`: Properly finalizes a communicator before destruction.
  - `ncclCommDestroy()`: Destroys a communicator and releases associated resources.

- **Memory Management**:
  - `ncclMemAlloc()`: Allocates device memory optimized for RCCL operations.
  - `ncclMemFree()`: Frees device memory allocated with RCCL memory management.

- **Collective Operations**:
  - `ncclGroupStart()`: Begins a group of collective operations for optimized execution.
  - `ncclGroupEnd()`: Ends a group of collective operations and launches them together.
  - `ncclAllReduce()`: Performs the AllReduce operation, reducing data from all ranks and distributing the result to all ranks.

- **Device API Concepts**:
  - **Kernel Fusion**: The Device API enables fusing computation kernels with collective operations, reducing memory traffic and latency.
  - **Gradient Clipping**: Demonstrates a common deep learning operation that benefits from fusion with communication.
  - **Memory Access Patterns**: Shows how fused operations reduce intermediate memory requirements.

- **HIP Kernel Programming**:
  - **Grid Configuration**: Uses configurable CTAs and threads per CTA for optimal performance.
  - **Grid Stride Loop**: Enables processing of arbitrary-sized arrays with fixed thread blocks.
  - **Mathematical Operations**: Demonstrates clipping functions using `fmaxf()` and `fminf()`.

- **Stream Management**: HIP streams are used for asynchronous execution and proper synchronization across multiple GPUs.

- **Data Types**: RCCL supports various data types including `ncclFloat` for single-precision floating-point operations.

## Demonstrated API Calls

### RCCL Collective Operations

- `ncclAllReduce`
- `ncclGroupStart`
- `ncclGroupEnd`

### RCCL Communicator Management

- `ncclCommInitAll`
- `ncclCommUserRank`
- `ncclCommCount`
- `ncclCommCuDevice`
- `ncclCommFinalize`
- `ncclCommDestroy`

### RCCL Memory Management

- `ncclMemAlloc`
- `ncclMemFree`

### HIP Runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipHostMalloc`
- `hipMalloc`
- `hipMemcpyAsync`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipSetDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`
- `hipLaunchKernel`
- `hipGetLastError`

### Data Types and Enums

- `ncclComm_t`
- `ncclDataType_t`
- `ncclRedOp_t`
- `ncclFloat`
- `ncclSum`
- `ncclResult_t`
- `hipStream_t`
