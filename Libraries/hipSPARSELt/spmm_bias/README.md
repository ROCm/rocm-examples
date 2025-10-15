# hipSPARSELt Sparse Matrix-Matrix Multiplication with Bias Addition Example

## Description

This example demonstrates how to perform *sparse matrix - dense matrix multiplication* with bias addition using
hipSPARSELt. On AMD Instinct™ MI300 GPUs it makes use of SMFMA (Sparse Matrix Fused Multiply Add) matrix instructions.
The calculation performed by this example is
$\mathbf{D} = \alpha \times \mathbf{A} \times \mathbf{B}^{\text{T}} + \beta \times \mathbf{C}$, where $\alpha$ and
$\beta$ are scalar values, $\mathbf{A}$ is a sparse matrix and $\mathbf{B}$, $\mathbf{C}$ and $\mathbf{D}$ are dense
matrices.

### Application flow

1. A HIP stream is created for later usage.
2. The hipSPARSELt library is initialized by obtaining a library handle.
3. $\mathbf{A}$ is created on the host and copied to the device:
    1. A structured (sparse) descriptor is created.
    2. A host buffer is allocated and initialized with random values.
    3. The buffer is copied to the device.
4. $\mathbf{B}$ and $\mathbf{C}$ are created on the host and copied to the device. For each matrix:
    1. A dense descriptor is created.
    2. A host buffer is allocated and initialized with random values.
    3. The buffer is copied to the device.
5. $\mathbf{D}$ is created on the device:
    1. A dense descripor is created.
    2. A device buffer is allocated and filled with zeroes.
6. The bias vecto is created on the host and copied to the device:
    1. A host buffer is allocated and initialized with random values.
    2. The buffer is copied to the device.
7. A descriptor for the matrix multiplication is created. $\mathbf{B}$ is marked for a transpose operation here.
8. The matrix multiplication is modified by setting attributes which enable the bias addition.
9. A matrix multiplication algorithm is automatically selected by hipSPARSELt.
10. A matrix multiplication plan is initialized.
11. A workspace buffer is allocated.
12. $\mathbf{A}$ is pruned using a 2:4 sparsity pattern.
13. The pruned $\mathbf{A}$ is compressed.
14. The matrix multiplication is performed.
15. $\mathbf{D}$ is copied back to the host.
16. All buffers, handles and descriptors are freed.

## Key APIs and Concepts

### hipSPARSELt

* hipSPARSELt is initialized by calling `hipsparseLtInit(hipsparseLtHandle_t*)` and is shut down by calling
  `hipsparseLtDestroy(hipsparseLtHandle_t*)`.
* A structured (sparse) matrix descriptor is obtained by calling `hipsparseLtStructuredDescriptorInit`.
* A dense matrix descriptor is obtained by calling `hipsparseLtDenseDescriptorInit`.
* A matrix descriptor of any type is freed by calling `hipsparseLtMatDescriptorDestroy`.
* A matrix multiplication descriptor is obtained by calling `hipsparseLtMatmulDescriptorInit`.
* A matrix multiplication descriptor's attributes are modified by calling `hipSparseLtMatmulDescSetAttribute`.
* An algorithm for matrix multiplication is selected by calling `hipsparseLtMatmulAlgSelectionInit`.
* A matrix multiplication plan is initialized by calling `hipsparseLtMatmulPlanInit` and freed by calling
  `hipsparseLtMatmulPlanDestroy`.
* The required amount of memory for hipSPARSELt's workspace is obtained by calling `hipsparseLtMatmulGetWorkspace`.
* A dense matrix is pruned by calling `hipsparseLtSpMMAPrune`.
* A pruning operation's success is queried by calling `hipsparseLtSpMMAPruneCheck`.
* The size of a compressed matrix is obtained by calling `hipsparseLtSpMMACompressedSize`.
* A pruned matrix is compressed by calling `hipsparseLtSpMMACompress`.
* A matrix multiplication $\mathbf{D} = \alpha \times \mathbf{A} \times \mathbf{B} + \beta \times \mathbf{C}$ is
  performed by calling `hipsparseLtMatmul`.

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
* `hipsparseLtMatmulDescSetAttribute`
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
