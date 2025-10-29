# RCCL Reduce Collective Communication

## Description

This example demonstrates the use of the RCCL library for Reduce collective communication operations.

The Reduce operation reduces data from all ranks to a single specified root rank:

$Output_{root} = Reduce(Input_0, Input_1, \ldots, Input_{N-1})$

where

- $Input_i$ is the input buffer from rank $i$ of size $k$ elements
- $Output_{root}$ is the output buffer at the root rank of size $k$ elements
- $N$ is the number of ranks in the communicator
- $root$ is the designated destination rank that receives the reduced result
- $Reduce$ is the reduction operation (sum, product, max, min, etc.)
- Only the root rank receives the reduced result; non-root ranks receive undefined output

## Application flow

1. Set up the number of ranks, detect available GPUs, and specify the root rank.
2. Initialize RCCL communicators for each GPU using `ncclCommInitAll()`.
3. Query communicator properties using `ncclCommUserRank()`, `ncclCommCount()`, and `ncclCommCuDevice()`.
4. Create HIP streams for each rank to enable asynchronous operations.
5. Allocate device memory for input and output buffers using `ncclMemAlloc()`.
6. Initialize host data with rank-specific patterns and copy to device memory.
7. Launch the Reduce operation using `ncclGroupStart()` and `ncclGroupEnd()` for optimized collective execution.
8. Perform the Reduce with `ncclReduce()`, reducing data from all ranks to the specified root rank using the sum operation.
9. Synchronize all streams to ensure completion of the collective operation.
10. Copy results back to host memory and verify the reduced data pattern only on the root rank.
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
  - `ncclReduce()`: Performs the Reduce operation, reducing data from all ranks to a single root rank.

- **Reduction Operations**: RCCL supports various reduction operations including `ncclSum` for summation, `ncclProd` for product, `ncclMax` for maximum, and `ncclMin` for minimum.

- **Root Rank Selection**: The root rank parameter specifies which rank will receive the final reduced result. All ranks must specify the same root rank.

- **Output Semantics**: Only the root rank receives the valid reduced result. Non-root ranks may have undefined output after the operation.

- **Stream Management**: HIP streams are used for asynchronous execution and proper synchronization across multiple GPUs.

- **Data Types**: RCCL supports various data types including `ncclFloat` for single-precision floating-point operations.

## Demonstrated API Calls

### RCCL Collective Operations

- `ncclReduce`
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
