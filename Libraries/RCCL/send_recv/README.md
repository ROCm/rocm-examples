# RCCL Point-to-Point Communication with Ring Topology

## Description

This example demonstrates RCCL point-to-point communication using Send and Recv operations with ring topology.

The ring communication pattern enables each rank to send data to the next rank and receive from the previous rank:

$Output_i = Input_{prev(i)}$ where $prev(i) = (i - 1 + N) \mod N$

where

- $Input_i$ is the input buffer from rank $i$ of size $k$ elements
- $Output_i$ is the output buffer at rank $i$ of size $k$ elements
- $N$ is the number of ranks in the communicator
- $prev(i)$ is the previous rank in the ring topology
- $next(i) = (i + 1) \mod N$ is the next rank in the ring
- Each rank receives data from its previous neighbor in the ring

## Application flow

1. Set up the number of ranks and detect available GPUs in the system.
2. Initialize RCCL communicators for each GPU using `ncclCommInitAll()`.
3. Query communicator properties using `ncclCommUserRank()`, `ncclCommCount()`, and `ncclCommCuDevice()`.
4. Create HIP streams for each rank to enable asynchronous operations.
5. Allocate device memory for send and receive buffers using `ncclMemAlloc()`.
6. Initialize host data with rank-specific patterns and copy to device memory.
7. Calculate ring topology: determine next and previous ranks for each rank.
8. Launch point-to-point communication using `ncclGroupStart()` and `ncclGroupEnd()` for deadlock avoidance.
9. Perform Send and Recv operations:
   - Each rank sends data to `next_rank` using `ncclSend()`.
   - Each rank receives data from `prev_rank` using `ncclRecv()`.
10. Synchronize all streams to ensure completion of point-to-point operations.
11. Copy results back to host memory and verify the ring communication pattern.
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

- **Point-to-Point Operations**:
  - `ncclSend()`: Sends data from the current rank to a specified destination rank.
  - `ncclRecv()`: Receives data from a specified source rank at the current rank.
  - **Non-blocking Operations**: Send and Recv operations are non-blocking and require synchronization.

- **Group Operations**:
  - `ncclGroupStart()`: Begins a group of operations for coordinated execution and deadlock prevention.
  - `ncclGroupEnd()`: Ends a group of operations and launches them together.
  - **Deadlock Avoidance**: Critical for ring patterns to prevent circular dependencies.

- **Ring Topology**:
  - **Circular Communication**: Data flows in a ring pattern: 0→1→2→…→N-1→0.
  - **Neighbor Calculation**: Each rank communicates only with immediate neighbors.
  - **Scalable Pattern**: Ring topology scales well with the number of ranks.

- **Synchronization Requirements**:
  - **Group Launch**: All Send/Recv operations must be launched in a single group.
  - **Stream Sync**: Proper synchronization ensures completion of all point-to-point operations.
  - **Verification**: Ring pattern verification ensures correct neighbor communication.

- **Stream Management**: HIP streams are used for asynchronous execution and proper synchronization across multiple GPUs.

- **Data Types**: RCCL supports various data types including `ncclFloat` for single-precision floating-point operations.

## Demonstrated API Calls

### RCCL Point-to-Point Operations

- `ncclSend`
- `ncclRecv`

### RCCL Group Operations

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
