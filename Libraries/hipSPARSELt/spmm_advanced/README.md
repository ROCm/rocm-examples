# hipSPARSELt Sparse Matrix-Matrix Multiplication example

## Description

This example demonstrates how to perform *sparse matrix - dense matrix multiplication* using hipSPARSELt. On AMD
Instinct™ MI300 GPUs, it makes use of SMFMA (Sparse Matrix Fused Multiply Add) matrix instructions. The calculation
performed by this example is
$\mathbf{D} = \alpha^{\text{T}} \times \mathbf{A} \times \mathbf{B}^{\text{T}} + \beta \times \mathbf{C}$, where
$\alpha$ is a vector, $\beta$ is a scalar value, $\mathbf{A}$ is a sparse matrix, and $\mathbf{B}$, $\mathbf{C}$, and
$\mathbf{D}$ are dense matrices. Additionally, a bias is added and the ReLU activation function is applied.

### Application flow

1. A HIP stream is created for later usage.
2. $\alpha$ is created on the host and copied to the device:
    1. A host buffer is allocated and initialized with random values.
    2. The buffer is copied to the device.
3. The hipSPARSELt library is initialized by obtaining a library handle.
4. $\mathbf{A}$ is created on the host and copied to the device:
    1. A structured (sparse) descriptor is created.
    2. A host buffer is allocated and initialized with random values.
    3. The buffer is copied to the device.
5. $\mathbf{B}$ and $\mathbf{C}$ are created on the host and copied to the device. For each matrix:
    1. A dense descriptor is created.
    2. A host buffer is allocated and initialized with random values.
    3. The buffer is copied to the device.
6. $\mathbf{D}$ is created on the device:
    1. A dense descriptor is created.
    2. A device buffer is allocated and filled with zeroes.
7. The bias vector is created on the host and copied to the device:
    1. A host buffer is allocated and initialized with random values.
    2. The buffer is copied to the device.
8. A descriptor for the matrix multiplication is created. $\mathbf{B}$ is marked for a transpose operation here.
9. The matrix multiplication's attributes are modified to include the multiplication of the $\alpha$ vector, bias
   addition, and ReLU activation.
10. A matrix multiplication algorithm is automatically selected by hipSPARSELt.
11. A matrix multiplication plan is initialized.
12. A workspace buffer is allocated.
13. $\mathbf{A}$ is pruned using a 2:4 sparsity pattern.
14. The pruned $\mathbf{A}$ is compressed.
15. The matrix multiplication is performed.
16. $\mathbf{D}$ is copied back to the host.
17. All buffers, handles, and descriptors are freed.

## Key APIs and concepts

### hipSPARSELt

* hipSPARSELt is initialized by calling `hipsparseLtInit(hipsparseLtHandle_t*)` and is closed by calling
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
