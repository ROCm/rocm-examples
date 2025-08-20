# rocWMMA Performance Half-Precision General Matrix Multiplication (HGEMM)

## Description

This example showcases an optimized half-precision GEMM implementation using advanced rocWMMA features for maximum performance. It demonstrates data reuse, latency hiding, and cooperative loading techniques to achieve high computational throughput.

The operation calculates the following product:

$D = \alpha \cdot A \cdot B + \beta \cdot C$

where:

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$

## Application flow

1. **Architecture-Specific Configuration**: Kernel parameters such as tile sizes and thread block dimensions are selected at compile time based on the target GPU architecture (GFX9 vs. GFX11) for optimal performance.
2. **Matrix Setup**: Host-side matrices are allocated and initialized. Input matrices A and B use half-precision (`float16_t`), while the computation and C/D matrices can use single-precision (`float32_t`) for better accuracy.
3. **Device Memory Management**: Device memory is allocated for all matrices, and the input data is copied from host to device.
4. **Kernel Configuration**:
    - Grid and block dimensions are determined based on the architecture-specific parameters.
    - The size of dynamic shared memory is calculated to accommodate double buffering of matrix tiles.
5. **Kernel Execution**: The `gemm_rocwmma_d` kernel is launched, which performs the following:
    - **Cooperative Loading**: Warps in a thread block collaborate to load large tiles of matrices A and B from global memory into LDS (shared memory).
    - **Double Buffering**: Two buffers in LDS are used in a ping-pong manner to overlap memory transfers with computation, effectively hiding memory latency.
    - **Matrix Multiplication**: rocWMMA's `mma_sync` function is called to perform the matrix multiplication on the tiles stored in LDS.
    - **Output**: The final result is calculated and stored back to global memory.
6. **Result Verification**: The output matrix D is copied back to the host and compared with a CPU-based reference implementation to ensure correctness.
7. **Cleanup**: All allocated device memory is freed.

## Key APIs and Concepts

- **Architecture-Specific Tuning**: The kernel uses different compile-time parameters (e.g., `rocwmma_m`, `rocwmma_n`, `tblock_x`) for GFX9 and GFX11 architectures to achieve the best performance on each platform.

- **Cooperative Loading**: `fragment_scheduler::coop_row_major_2d` is used to coordinate the loading of data from global memory by all warps in a thread block. This ensures efficient use of memory bandwidth.

- **Double Buffering in LDS**:
  - The kernel allocates two buffers in dynamic shared memory (`HIP_DYNAMIC_SHARED`).
  - It prefetches the next required matrix tiles into one buffer while the current tiles are being processed from the other.
  - This overlapping of data movement and computation is a key technique for hiding memory latency.

- **Data Layout Transformations**:
  - `apply_data_layout_t` and `apply_transpose_t` are rocWMMA transforms used to change the memory layout of fragments. This is useful for optimizing data access patterns when moving data between global memory, LDS, and registers.

- **Mixed Precision**: The example uses half-precision (`float16_t`) for input matrices to reduce memory bandwidth and storage requirements, while performing the accumulation in single-precision (`float32_t`) to maintain numerical accuracy.

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

- `rocwmma::float16_t`
- `rocwmma::float32_t`
- `rocwmma::row_major`
- `rocwmma::col_major`
- `HIP_DYNAMIC_SHARED`
