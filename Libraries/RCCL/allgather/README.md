# RCCL AllGather Collective Communication

## Description

This example demonstrates the use of the RCCL library for AllGather collective communication operations.

The AllGather operation collects data from all ranks and distributes the concatenated result to every rank:

$Output_i = [Input_0, Input_1, \ldots, Input_{N-1}]$

where

- $Input_i$ is the input buffer from rank $i$ of size $k$ elements
- $Output_i$ is the output buffer at rank $i$ of size $N \times k$ elements
- $N$ is the number of ranks in the communicator
- Each rank receives the complete concatenated data from all ranks

## Application flow

1. Set up the number of ranks and detect available GPUs in the system.
2. Initialize RCCL communicators for each GPU using `ncclCommInitAll()`.
3. Query communicator properties using `ncclCommUserRank()`, `ncclCommCount()`, and `ncclCommCuDevice()`.
4. Create HIP streams for each rank to enable asynchronous operations.
5. Allocate device memory for input and output buffers using `ncclMemAlloc()`.
6. Initialize host data with rank-specific patterns and copy to device memory.
7. Launch the AllGather operation using `ncclGroupStart()` and `ncclGroupEnd()` for optimized collective execution.
8. Perform the AllGather with `ncclAllGather()`, gathering data from all ranks to all ranks.
9. Synchronize all streams to ensure completion of the collective operation.
10. Copy results back to host memory and verify the concatenated data pattern.
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
  - `ncclAllGather()`: Performs the AllGather operation, gathering data from all ranks and distributing the concatenated result to all ranks.

- **Stream Management**: HIP streams are used for asynchronous execution and proper synchronization across multiple GPUs.

- **Data Types**: RCCL supports various data types including `ncclFloat` for single-precision floating-point operations.

## Demonstrated API Calls

### RCCL Collective Operations

- `ncclAllGather`
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
