# rocWMMA Performance Single-Precision General Matrix Multiplication (SGEMM)

## Description

This example demonstrates an optimized single-precision GEMM implementation featuring advanced data reuse and prefetching strategies for maximum FP32 performance. It showcases techniques to achieve high computational throughput while maintaining numerical precision.

The operation calculates the following product:

$D = \alpha \cdot A \cdot B + \beta \cdot C$

where:

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$

## Application flow

1. **Device Capability Check**: The application checks if the device supports single-precision (FP32) operations.
2. **Matrix Setup**: Host-side matrices (A, B, C, D) are allocated and initialized with random data.
3. **Device Memory Management**: Device memory is allocated for all matrices. Input matrices are copied from host to device.
4. **Kernel Configuration**:
    - Grid and block dimensions are calculated based on matrix and tile sizes.
    - Dynamic shared memory size is calculated for double buffering.
5. **Kernel Execution**: The `sgemm_rocwmma_d` kernel is launched with the configured parameters. The kernel performs the following steps:
    - **Cooperative Loading**: Warps within a thread block cooperate to prefetch matrix tiles from global memory into LDS (shared memory).
    - **Double Buffering**: Two buffers in LDS are used in a ping-pong fashion to overlap data fetching with computation, hiding memory latency.
    - **Matrix Multiplication**: rocWMMA's `mma_sync` is used to perform the matrix multiplication on the tiles loaded from LDS.
    - **Final Output**: The result is scaled by alpha and beta, combined with the C matrix, and stored back to global memory.
6. **Result Verification**: The output matrix D is copied back to the host and compared against a CPU-based reference implementation to verify correctness.
7. **Cleanup**: All device memory is freed.

## Key APIs and Concepts

- **Cooperative Loading**: Multiple warps within a thread block work together to load a larger tile of data from global memory into LDS. This is managed by `fragment_scheduler::coop_row_major_2d`, which coordinates the memory access across the warps.

- **Double Buffering in LDS**:
  - Two separate buffers are allocated in dynamic shared memory.
  - While the compute units are processing data from one buffer, the next set of data is prefetched into the other buffer.
  - This technique, often called "ping-pong buffering," helps to hide the latency of global memory access.

- **Data Layout Transformations**:
  - `apply_data_layout_t`: A rocWMMA transform used to change the data layout of a fragment, for example, from row-major in global memory to column-major in LDS for better memory access patterns.
  - `apply_transpose_t`: A rocWMMA transform to transpose a matrix fragment.

- **rocWMMA Fragments**:
  - `rocwmma::fragment`: Represents a piece of a matrix held in registers by a single warp. Different fragment types are defined for matrices A, B, and the accumulator (C/D).
  - `rocwmma::load_matrix_sync()`: Loads data from global or shared memory into a fragment.
  - `rocwmma::store_matrix_sync()`: Stores data from a fragment to global or shared memory.
  - `rocwmma::mma_sync()`: Performs the matrix multiplication and accumulation operation on fragments.

- **Workgroup Synchronization**: `rocwmma::synchronize_workgroup()` is used to ensure that all warps in a thread block have completed a certain stage (e.g., writing to LDS) before proceeding to the next.

## Demonstrated API Calls

### rocWMMA

- `rocwmma::fragment`
- `rocwmma::load_matrix_sync`
- `rocwmma::store_matrix_sync`
- `rocwmma::mma_sync`
- `rocwmma::fill_fragment`
- `rocwmma::synchronize_workgroup`
- `rocwmma::apply_data_layout_t`
- `rocwmma::apply_transpose_t`
- `rocwmma::fragment_scheduler::coop_row_major_2d`

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
- `hipGetDevice`
- `hipGetDeviceProperties`
- `hipEventCreate`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipEventElapsedTime`
- `hipEventDestroy`

## Data Types and Enums

- `rocwmma::float32_t`
- `rocwmma::row_major`
- `rocwmma::col_major`
- `HIP_DYNAMIC_SHARED`
