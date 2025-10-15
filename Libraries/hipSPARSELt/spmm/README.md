# hipSPARSELt Sparse Matrix-Matrix Multiplication Example

## Description

This example demonstrates how to perform *sparse matrix - dense matrix multiplication* using hipSPARSELt. On AMD
Instinct™ MI300 GPUs it makes use of SMFMA (Sparse Matrix Fused Multiply Add) matrix instructions. The calculation
performed by this example is $D = \alpha \times \mathbf{A} \times \mathbf{B}^{\text{T}} + \beta \times \mathbf{C}$,
where $\alpha$ and $\beta$ are scalar values, $\mathbf{A}$ is a sparse matrix and $\mathbf{B}$, $\mathbf{C}$ and
$\mathbf{D}$ are dense matrices.

### Application flow

1. A HIP stream is created for later usage.
2. The hipSPARSELt library is initialized by obtaining a library handle.
3. $\mathbf{A}$ is created on the host and copied to the device:
    1. A structured (sparse) descriptor is created.
    2. A host buffer is allocated and initialized with random values.
    3. The buffer is copied to the device.
4. $\mathbf{B}$ and $$\mathbf{C}$ are created on the host and copied to the device. For each matrix:
    1. A dense descriptor is created.
    2. A host buffer is allocated and initialized with random values.
    3. The buffer is copied to the device.
5. $\mathbf{D}$ is created on the device:
    1. A dense descripor is created.
    2. A device buffer is allocated and filled with zeroes.
6. A descriptor for the matrix multiplication is created. $\mathbf{B}$ is marked for a transpose operation here.
7. A matrix multiplication algorithm is automatically selected by hipSPARSELt.
8. A matrix multiplication plan is initialized.
9. A workspace buffer is allocated.
10. $\mathbf{A}$ is pruned using a 2:4 sparsity pattern.
11. The pruned $\mathbf{A}$ is compressed.
12. The matrix multiplication is performed.
13. $\mathbf{D}$ is copied back to the host.
14. All buffers, handles and descriptors are freed.

## Key APIs and Concepts

### hipSPARSELt

* hipSPARSELt is initialized by calling `hipsparseLtInit(hipsparseLtHandle_t*)` and is shut down by calling
  `hipsparseLtDestroy(hipsparseLtHandle_t*)`.
* A structured (sparse) matrix descriptor is obtained by calling `hipsparseLtStructuredDescriptorInit` which takes the
  following parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `hipsparseLtMatDescriptor_t*`: The matrix descriptor handle representing the descriptor after the call.
  * `int64_t`: The number of rows in the matrix.
  * `int64_t`: The number of columns in the matrix.
  * `int64_t`: The leading dimension. >= rows for a column-major memory layout, >= colums for a row-major layout.
  * `uint32_t`: Memory alignment (not used for AMD targets)
  * `hipDataType`: The datatype used by the matrix.
  * `hipsparseOrder_t`: The memory layout, either `HIPSPARSE_ORDER_COL` or `HIPSPARSE_ORDER_ROW`.
  * `hipsparseLtSparsity_t`: The sparsity ratio. `HIPSPARSE_SPARSITY_50_PERCENT` is the only valid value.
* A dense matrix descriptor is obtained by calling `hipsparseLtDenseDescriptorInit` which uses the same input parameters
  as `hipsparseLtStructuredDescriptorInit` except for the omitted `hipsparseLtSparsity_t` parameter.
* Matrix descriptors are freed by calling `hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t*)`.
* A matrix multiplication descriptor is obtained by calling `hipsparseLtMatmulDescriptorInit` which takes the following
  parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `hipsparseLtMatmulDescriptor_t*`: The matrix multiplication descriptor handle representing the descriptor after the
    call.
  * `hipsparseOperation_t`: Whether to transpose $\mathbf{A}$. Valid values are
    `HIPSPARSE_OPERATION_NON_TRANSPOSE` or `HIPSPARSE_OPERATION_TRANSPOSE`.
  * `hipsparseOperation_t`: Whether to transpose $\mathbf{B}$. Valid values are
    `HIPSPARSE_OPERATION_NON_TRANSPOSE` or `HIPSPARSE_OPERATION_TRANSPOSE`.
  * `const hipsparseLtMatDescriptor_t*`: $\mathbf{A}$'s descriptor.
  * `const hipsparseLtMatDescriptor_t*`: $\mathbf{B}$'s descriptor.
  * `const hipsparseLtMatDescriptor_t*`: $\mathbf{C}$'s descriptor.
  * `const hipsparseLtMatDescriptor_t*`: $\mathbf{D}$'s descriptor.
  * `hipsparseLtComputetype_t`: The compute precision mode. Valid values are:
    * `HIPSPARSELT_COMPUTE_16F`: 16-bit floating-point precision. CUDA backend only.
    * `HIPSPARSELT_COMPUTE_32I`: 32-bit integer precision.
    * `HIPSPARSELT_COMPUTE_32F`: 32-bit floating-point precision. HIP backend only.
    * `HIPSPARSELT_COMPUTE_TF32`: 32-bit floating point value are rounded to TF32 before the computation. CUDA backend
      only.
    * `HIPSPARSELT_COMPUTE_TF32_FAST`: 32-bit floating point value are truncated to TF32 before the computation. CUDA
      backend only.
* The algorithm for matrix multiplication is selected by calling `hipsparseLtMatmulAlgSelectionInit` which takes the
  following parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `hipsparseLtMatmulAlgSelection_t*`: The algorithm handle representing the selected algorithm after the call.
  * `const hipsparseLtMatmulDescriptor_t*`: A matrix multiplication descriptor.
  * `hipsparseLtMatmulAlg_t`. The algorithm used to perform the matrix multiplication. `HIPSPARSELT_MATMUL_ALG_DEFAULT`
    is the only valid value.
* The matrix multiplication plan is initialized by calling `hipsparseLtMatmulPlanInit` which takes the following
  parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `hipsparseLtMatmulPlan_t*`: The plan handle representing the plan after the call.
  * `const hipsparseLtMatmulDescriptor_t*`: A matrix multiplication descriptor.
  * `const hipsparseLtMatmulAlgSelection_t*`: A selected algorithm.
* The matrix multiplication plan is freed by calling `hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t*)`.
* The required amount of memory for hipSPARSELt's workspace is obtained by calling `hipsparseLtMatmulGetWorkspace` which
  takes the following parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `const hipsparseLtMatmulPlan_t*`: A matrix multiplication plan.
  * `size_t*`: The workspace size in bytes.
* A dense matrix is pruned by calling `hipsparseLtSpMMAPrune` which takes the following parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `const hipsparseLtMatmulDescriptor_t*`: A matrix multiplication descriptor.
  * `const void*`: A pointer to a dense matrix.
  * `void*`: A pointer to the resulting pruned matrix.
  * `hipsparseLtPruneAlg_t`: The pruning algorithm. Valid values are:
    * `HIPSPARSELT_PRUNE_SPMMA_TILE`: Zero out eight elements in a 4x4 tile, nonzero elements have the maximum L1-norm
      value in all combinations in the tile. Exactly two elements in each row and column.
    * `HIPSPARSELT_PRUNE_SPMMA_STRIP`: Zero out two elements in a 1x4 strip, nonzero elements have the maximum L1-norm
      value in all combinations in the strip.
  * `hipStream_t`: The stream to perform the pruning operation on.
* A pruning operation's success is queried by calling `hipsparseLtSpMMAPruneCheck` which takes the following parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `const hipsparseLtMatmulDescriptor_t*`: A matrix multiplication descriptor.
  * `const void*`: A pointer to a pruned matrix.
  * `int*`: Whether or not the pruning operation succeeded. `0` if correct, `1` if wrong.
  * `hipStream_t`: The stream to perform the validation operation on.
* The size of a compressed matrix is obtained by calling `hipsparseLtSpMMACompressedSize` which takes the following
  parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `const hipsparseLtMatmulPlan_t*`: A matrix multiplication plan.
  * `size_t*`: The compressed matrix size in bytes.
  * `size_t*`: The temporary buffer's (required for the compression operation) size in bytes.
* A pruned matrix is compressed by calling `hipsparseLtSpMMACompress` which takes the following parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `const hipsparseLtMatmulPlan_t*`: A matrix multiplication plan.
  * `const void*`: A pointer to a pruned matrix.
  * `void*`: A pointer to the resulting compressed matrix.
  * `void*`: A pointer to the temporary buffer required for the compression operation.
  * `hipStream_t`: The stream to perform the compression operation on.
* A matrix multiplication $D = \alpha \times \mathbf{A} \times \mathbf{B} + \beta \times \mathbf{C}$ is performed by
  calling `hipsparseLtMatmul` which takes the following parameters:
  * `const hipsparseLtHandle_t*`: The library handle.
  * `const hipsparseLtMatmulPlan_t*`: A matrix multiplication plan.
  * `const void*`: A pointer to the scalar $\alpha$.
  * `const void*`: A pointer to the sparse matrix $\mathbf{A}$.
  * `const void*`: A pointer to the dense matrix $\mathbf{B}$.
  * `const void*`: A pointer to the scalar $\beta$.
  * `const void*`: A pointer to the dense matrix $\mathbf{C}$.
  * `void*`: A pointer to the dense matrix $\mathbf{D}$.
  * `void*`: A pointer to the workspace buffer.
  * `hipStream_t*`: An array of streams to perform the matrix multiplication operation on.
  * `int32_t`: The number of streams in the array.

## Used API surface

### hipSPARSELt

#### Types

* `hipsparseLtHandle_t`
* `hipsparseLtMatDescriptor_t`
* `hipsparseLtMatmulAlgSelection_t`
* `hipsparseLtMatmulDescriptor_t`
* `hipsparseLtMatmulPlan_t`

#### Functions

* `hipsparseLtDenseDescriptorInit`
* `hipsparseLtDestroy`
* `hipsparseLtInit`
* `hipsparseLtMatDescriptorDestroy`
* `hipsparseLtMatmul`
* `hipsparseLtMatmulAlgSelectionInit`
* `hipsparseLtMatmulDescriptorInit`
* `hipsparseLtMatmulGetWorkspace`
* `hipsparseLtMatmulPlanDestroy`
* `hipsparseLtMatmulPlanInit`
* `hipsparseLtSpMMACompress`
* `hipsparseLtSpMMACompressedSize`
* `hipsparseLtSpMMAPrune`
* `hipsparseLtSpMMAPruneCheck`
* `hipsparseLtStructuredDescriptorInit`

## HIP runtime

* `hipFree`
* `hipMalloc`
* `hipMemcpy`
* `hipMemset`
* `hipStreamCreate`
* `hipStreamDestroy`
* `hipStreamSynchronize`
