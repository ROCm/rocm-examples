# CK Tile Programming Model: MoE Sorting example

## Description

This example demonstrates how to implement the MoE sorting kernel using the CK Tile programming model. This kernel is
commonly used in MoE models before launching the fused MoE GEMM block. The input and weight form a `token × topk` 2D
matrix. The operation rearranges the input weight IDs into different experts and feeds them into the fused MoE GEMM
kernel.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers are created for all tensors on the host.
3. The input tensors are initialized with random values.
4. Buffers are created for all tensors on the device.
5. The input tensors are copied to the device.
6. The workspace size is obtained.
7. Depending on the chosen parameters one of CK Tile's `MoeSortingMultiPhaseKernel`s or `MoeSortingKernel` is
   instantiated and launched on the device.
8. If validation is enabled, the results are compared against an implementation using CK Tile's `reference_moe_sorting`
   function.
9. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of two key architectural components:

* A **problem** combines data types with the shape configuration. In this example the problem is set to CK Tile's
  `MoeSortingProblemMp`, `MoeSortingProblemEx` or `MoeSortingProblem`, depending on the chosen parameters.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `MoeSortingMultiPhaseKernel` or `MoeSortingKernel` are used, depending on the chosen parameters.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::MoeSortingKernel`
* `ck_tile::MoeSortingMultiPhaseKernel_P0`
* `ck_tile::MoeSortingMultiPhaseKernel_P1`
* `ck_tile::MoeSortingMultiPhaseKernel_P2`
* `ck_tile::MoeSortingMultiPhaseKernel_P3`
* `ck_tile::MoeSortingMultiPhaseKernel_P23`
* `ck_tile::MoeSortingProblem`
* `ck_tile::MoeSortingProblemMp`
* `ck_tile::number`
* `ck_tile::remove_cvref_t`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::get_smem_capacity`
* `ck_tile::integer_divide_ceil`
* `ck_tile::integer_least_multiple`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::moe_sorting_get_sub_token`
* `ck_tile::moe_sorting_get_workspace_size`
* `ck_tile::reference_moe_sorting`

### HIP runtime

#### Types

* `dim3`
