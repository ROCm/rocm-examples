# rocWMMA Simple Deep Learning Recommendation Model (DLRM) Example

## Description

This example demonstrates the use of rocWMMA for a key computation in Deep Learning Recommendation Models (DLRM): the feature interaction step. It performs pairwise dot products of embedding vectors, a common operation in recommendation systems, showcasing how rocWMMA can be applied to machine learning workloads.

The operation computes pairwise interactions between embedding vectors. The input is a batch of embedding tables, and the output consists of the original MLP features followed by the lower triangular part of the interaction matrix.

## Application flow

1. **Input Data**: A batch of input embeddings is created on the host, using half-precision (`float16_t`) for memory efficiency.
2. **Device Memory Allocation**: Device memory is allocated for the input embeddings, the output (MLP features + interactions), and an intermediate accumulation buffer.
3. **Kernel Launch**: The `dlrm_dot_fwd` kernel is launched with a 3D grid to handle the batch of embeddings.
4. **MLP Feature Copy**: The first part of the kernel copies the bottom MLP features directly to the output buffer.
5. **Feature Interaction**:
    - The kernel computes the dot product of every pair of embedding vectors. This is formulated as a matrix multiplication of the embedding matrix with its transpose ($E \cdot E^T$).
    - rocWMMA is used to perform this matrix multiplication efficiently.
    - The result is stored in a temporary accumulation buffer.
6. **Triangular Output**: A final step in the kernel reads the accumulation buffer and stores the lower triangular part of the interaction matrix to the final output buffer. This avoids storing redundant information.
7. **Result Verification**: The output is copied back to the host and compared against a CPU reference implementation.
8. **Cleanup**: Device memory is freed.

## Key APIs and Concepts

- **DLRM Feature Interaction**: The core of this example is the computation of pairwise feature interactions, which is a fundamental part of DLRMs. This is efficiently implemented as a matrix multiplication ($E \cdot E^T$).

- **Mixed Precision**: The example uses half-precision (`float16_t`) for the input embeddings to save memory and bandwidth, while the accumulation is done in single-precision (`float32_t`) to maintain numerical stability and accuracy.

- **Batch Processing**: The kernel is designed to process a batch of inputs simultaneously. A 3D grid is used, where the z-dimension of the grid corresponds to the batch index.

- **rocWMMA for Self-Attention**: The computation of $E \cdot E^T$ is a form of self-attention, a common pattern in many machine learning models. This example demonstrates how rocWMMA can be used to accelerate such operations.

- **`rocwmma::MappingUtil`**: This utility is used to simplify the mapping of threads to matrix coordinates, which can be complex in tiled matrix operations.

## Demonstrated API Calls

### rocWMMA

- `rocwmma::fragment`
- `rocwmma::load_matrix_sync`
- `rocwmma::store_matrix_sync`
- `rocwmma::mma_sync`
- `rocwmma::fill_fragment`
- `rocwmma::synchronize_workgroup`
- `rocwmma::MappingUtil`

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
-- `hipFree`
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
