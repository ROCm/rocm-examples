# RCCL Buffer Registration for Performance Optimization

## Description

This example demonstrates RCCL buffer registration optimization for repeated collective operations.

Buffer registration optimizes performance by pre-registering memory buffers for multiple operations:

$Output_i = \sum_{j=0}^{N-1} Input_{j}^{(iter)}$ for each iteration

where

- $Input_{j}^{(iter)}$ is the input buffer from rank $j$ at iteration $iter$
- $Output_i$ is the output buffer at rank $i$ after AllReduce operation
- $N$ is the number of ranks in the communicator
- $Buffers$ are registered once and reused across multiple iterations
- Registration eliminates per-iteration memory management overhead

## Application flow

1. Set up the number of ranks, total elements, number of layers, and training iterations.
2. Detect available GPUs and configure the number of ranks.
3. Initialize RCCL communicators for each GPU using `ncclCommInitAll()`.
4. Query communicator properties using `ncclCommUserRank()`, `ncclCommCount()`, and `ncclCommCuDevice()`.
5. Create HIP streams for each rank to enable asynchronous operations.
6. Allocate device memory for multiple layers using `ncclMemAlloc()`.
7. Register all buffers with communicators using `ncclCommRegister()` for optimization.
8. Initialize host data with layer-specific patterns and copy to device memory.
9. Perform multiple training iterations, each containing:
   - Group all AllReduce operations using `ncclGroupStart()` and `ncclGroupEnd()`.
   - Execute `ncclAllReduce()` for each layer using registered buffers.
   - Synchronize streams to ensure completion of all collective operations.
10. Copy results back to host memory and verify the buffer registration benefits.
11. Deregister all buffers using `ncclCommDeregister()` before cleanup.
12. Clean up resources in the proper order: destroy streams, free memory, finalize and destroy communicators.

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

- **Buffer Registration**:
  - `ncclCommRegister()`: Registers a memory buffer with a communicator for optimized repeated operations.
  - `ncclCommDeregister()`: Deregisters a previously registered memory buffer.
  - **Performance Benefits**: Eliminates per-iteration registration overhead for repeated operations.
  - **Training Loop Optimization**: Ideal for deep learning training loops with repeated gradient synchronizations.

- **Collective Operations**:
  - `ncclGroupStart()`: Begins a group of collective operations for optimized execution.
  - `ncclGroupEnd()`: Ends a group of collective operations and launches them together.
  - `ncclAllReduce()`: Performs the AllReduce operation, reducing data from all ranks and distributing the result to all ranks.

- **Multi-Layer Pattern**:
  - **Layer Isolation**: Each gradient layer is registered and managed independently.
  - **Memory Efficiency**: Reduces allocation overhead for multiple-layer scenarios.
  - **Batched Operations**: Multiple AllReduce operations are grouped for better performance.

- **Stream Management**: HIP streams are used for asynchronous execution and proper synchronization across multiple GPUs.

- **Data Types**: RCCL supports various data types including `ncclFloat` for single-precision floating-point operations.

## Demonstrated API Calls

### RCCL Buffer Registration

- `ncclCommRegister`
- `ncclCommDeregister`

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
