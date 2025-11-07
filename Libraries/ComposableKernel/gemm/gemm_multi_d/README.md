# CK Tile Programming Model: Multiple D GEMM example

## Description

This example demonstrates how to perform a Multiple D GEMM operation with the CK Tile programming model.

### Supported architectures

The example is supported for the following architectures:

* `gfx908`
* `gfx90a`
* `gfx942`
* `gfx950`

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for the input matrices and the output matrix are created on the host.
3. The input matrices are initialized with random values.
4. Buffers for the input matrices and the output matrix are created on the device.
5. The input matrices are copied to the device, the output matrix is initialized to `0` on the device.
6. CK Tile's `GemmKernel` is instantiated and launched on the device.
7. If validation is enabled, the results are compared against CK Tile's `reference_gemm_multiple_d` function.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of four key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this example it is set to CK Tile's
  `TileGemmShape`.
* A **problem** combines data types with the shape configuration. In this example the problem is set to CK Tile's
  `UniversalGemmPipelineProblem`.
* A **pipeline** schedules the sequence of operations for a kernel, such as the data loading, computation, and
  storage phases. In this example the pipeline is set to `BaseGemmPipelineAgBgCrCompV3`.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `GemmKernelMultiD` is used.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::bool_constant`
* `ck_tile::CShuffleEpilogue`
* `ck_tile::CShuffleEpilogueProblem`
* `ck_tile::DeviceMem`
* `ck_tile::element_wise::PassThrough`
* `ck_tile::FillUniformDistribution`
* `ck_tile::BaseGemmPipelineAgBgCrCompV3`
* `ck_tile::BaseGemmPipelineAgBgCrCompV4`
* `ck_tile::BaseGemmPipelineAgBgCrMem`
* `ck_tile::GemmMultiDHostArgs`
* `ck_tile::GemmKernelMultiD`
* `ck_tile::GemmPipelineProblem`
* `ck_tile::GemmPipelineScheduler::Interwave`
* `ck_tile::GemmPipelineScheduler::Intrawave`
* `ck_tile::GemmSpatiallyLocalTilePartitioner`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::integral_constant`
* `ck_tile::memory_operation_enum`
* `ck_tile::sequence`
* `ck_tile::stream_config`
* `ck_tile::TailNumber`
* `ck_tile::tensor_layout::gemm::ColumnMajor`
* `ck_tile::tensor_layout::gemm::RowMajor`
* `ck_tile::TileGemmShape`
* `ck_tile::TileGemmTraits`
* `ck_tile::TileGemmUniversalTraits`
* `ck_tile::UniversalGemmPipelineProblem`

#### Functions

* `ck_tile::check_err`
* `ck_tile::host_tensor_descriptor`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::ck_tile::reference_gemm_multiple_d`
* `ck_tile::remove_cvref_t`
* `ck_tile::static_for`

### HIP runtime

#### Host symbols

* `dim3`
