# RCCL Broadcast Collective Communication

## Description

This example demonstrates the use of the RCCL library for Broadcast collective communication operations.

The Broadcast operation distributes data from a root rank to all ranks in the communicator:

$Output_i = Input_{root}$ for all ranks $i = 0, 1, \ldots, N-1$

where

- $Input_{root}$ is the input buffer from the designated root rank of size $k$ elements
- $Output_i$ is the output buffer at rank $i$ of size $k$ elements
- $root$ is the designated source rank that broadcasts the data
- $N$ is the number of ranks in the communicator
- All ranks receive identical data from the root rank

## Application flow

1. Set up the number of ranks, detect available GPUs, and specify the root rank.
2. Initialize RCCL communicators for each GPU using `ncclCommInitAll()`.
3. Query communicator properties using `ncclCommUserRank()`, `ncclCommCount()`, and `ncclCommCuDevice()`.
4. Create HIP streams for each rank to enable asynchronous operations.
5. Allocate device memory for input and output buffers using `ncclMemAlloc()`.
6. Initialize host data with meaningful data on the root rank and zeros on non-root ranks, then copy to device memory.
7. Launch the Broadcast operation using `ncclGroupStart()` and `ncclGroupEnd()` for optimized collective execution.
8. Perform the Broadcast with `ncclBroadcast()`, distributing data from the root rank to all ranks.
9. Synchronize all streams to ensure completion of the collective operation.
10. Copy results back to host memory and verify that all ranks received the root's data.
11. Clean up resources in the proper order: destroy streams, free memory, finalize and destroy communicators.

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
  - `ncclBroadcast()`: Performs the Broadcast operation, distributing data from a root rank to all ranks in the communicator.

- **Root Rank Selection**: The root rank parameter specifies which rank's data will be broadcast to all other ranks. All ranks must specify the same root rank.

- **Stream Management**: HIP streams are used for asynchronous execution and proper synchronization across multiple GPUs.

- **Data Types**: RCCL supports various data types including `ncclFloat` for single-precision floating-point operations.

## Demonstrated API Calls

### RCCL Collective Operations

- `ncclBroadcast`
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
- `ncclFloat`
- `ncclResult_t`
