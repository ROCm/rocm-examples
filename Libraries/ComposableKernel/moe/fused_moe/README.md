# CK Tile Programming Model: Fused-MoE example

## Description

This example shows how to implement the fused MoE block operator using the CK Tile programming model. This is a
scatter/gather-group-GEMM based solution, similar to that of
[vLLM's MoE implementation](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py).

The algorithm implemented here utilizes more kernel fusion to boost performance. Compared to the vLLM solution, this
achieves a 1.5~2x performance boost. In addition, it uses no workspace memory und requires less kernel instances which
improves maintainability.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers are created for all tensors on the host and initialized with random values.
3. The weight tensors are permuted.
4. Buffers are created for all tensors on the device.
5. The input tensors are copied to the device.
6. Depending on the chosen parameters one of CK Tile's `MoeSortingMultiPhaseKernel`s is instantiated and launched on the
   device. The MoE loop is transformed from token-by-token to expert-by-expert to make sure every workgroup is working
   for a single expert. Additionally, this operation initializes the output tensor to `0`.
7. CK Tile's `FusedMoeGemmKernel` is instantiated and launched on the device.
8. If validation is enabled, the results are compared against an implementation using CK Tile's `reference_moe_sorting`
   and `reference_fused_moe` functions.
9. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of four key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this example it is set to CK Tile's
  `FusedMoeGemmShape`.
* A **problem** combines data types with the shape configuration. In this example the problem is set to CK Tile's
  `FusedMoeGemmPipelineProblem`.
* A **pipeline** schedules the sequence of operations for a kernel, such as the data loading, computation, and 
  storage phases. In this example the pipeline is set to `FusedMoeGemmPipeline_FlatmmUk`.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `FusedMoeGemmKernel` is used.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::DeviceMem`
* `ck_tile::element_wise::FastGeluAsm`
* `ck_tile::element_wise::Gelu`
* `ck_tile::element_wise::Silu`
* `ck_tile::FillNormalDistribution`
* `ck_tile::FillStepRange`
* `ck_tile::FillUniformDistribution`
* `ck_tile::fp16_t`
* `ck_tile::FusedMoeGemmHostArgs`
* `ck_tile::FusedMoeGemmKernel`
* `ck_tile::FusedMoeGemmPipelineProblem`
* `ck_tile::FusedMoeGemmPipeline_FlatmmUk`
* `ck_tile::FusedMoeGemmShape`
* `ck_tile::FusedMoeGemmTilePartitioner_Linear`
* `ck_tile::FusedMoeGemmTraits`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::int8_t`
* `ck_tile::long_index_t`
* `ck_tile::number`
* `ck_tile::remove_cvref_t`
* `ck_tile::sequence`
* `ck_tile::stream_config`


#### Functions

* `ck_tile::check_err`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::moe_sorting_get_workspace_size`
* `ck_tile::reference_fused_moe`
* `ck_tile::reference_moe_sorting`
* `ck_tile::reference_permute`

### HIP runtime

#### Types

* `dim3`