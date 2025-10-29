# RCCL Gradient AllReduce for Distributed Training

## Description

This example demonstrates RCCL AllReduce operations in a distributed deep learning training scenario with multiple gradient layers.

The AllReduce operation synchronizes gradients across all ranks for distributed data parallel training:

$Output_i = \sum_{j=0}^{N-1} Gradient_{i,j}$ for each layer $l$

where

- $Gradient_{i,j,l}$ is the gradient tensor for layer $l$ on rank $i$ from rank $j$
- $Output_{i,l}$ is the synchronized gradient tensor for layer $l$ on rank $i$
- $N$ is the number of ranks in the communicator
- $L$ is the number of gradient layers
- Each rank receives the sum of gradients from all ranks for each layer

## Application flow

1. Set up the number of ranks, total elements, number of layers, and training iterations.
2. Detect available GPUs and configure the number of ranks.
3. Initialize RCCL communicators for each GPU using `ncclCommInitAll()`.
4. Query communicator properties using `ncclCommUserRank()`, `ncclCommCount()`, and `ncclCommCuDevice()`.
5. Create HIP streams for each rank to enable asynchronous operations.
6. Allocate device memory for multiple gradient layers (inputs and outputs) using `ncclMemAlloc()`.
7. Initialize host data with layer-specific gradient patterns and copy to device memory.
8. Perform training iterations, each containing:
   - Group all AllReduce operations using `ncclGroupStart()` and `ncclGroupEnd()`.
   - Execute `ncclAllReduce()` for each layer to synchronize gradients across all ranks.
   - Synchronize streams to ensure completion of all collective operations.
9. Copy results back to host memory and verify the gradient synchronization for each layer.
10. Clean up resources in the proper order: destroy streams, free memory, finalize and destroy communicators.

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

- **Distributed Training Pattern**:
  - **Multi-Layer Gradients**: Simulates realistic neural network training with multiple gradient tensors.
  - **Training Iterations**: Demonstrates repeated gradient synchronization across training steps.
  - **Layer Isolation**: Each layer's gradients are synchronized independently.
  - **DDP Pattern**: Follows Distributed Data Parallel training methodology.

- **Performance Optimization**:
  - **Group Operations**: Batching multiple AllReduces for better network utilization.
  - **Asynchronous Execution**: Using HIP streams for overlapping computation and communication.
  - **Memory Management**: Efficient allocation patterns for multiple gradient buffers.

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

### Data Types and Enums

- `ncclComm_t`
- `ncclDataType_t`
- `ncclRedOp_t`
- `ncclFloat`
- `ncclSum`
- `ncclResult_t`
- `hipStream_t`
