# CK Tile Programming Model: Grouped GEMM example

## Description

This example demonstrates how to perform a GEMM operation with the CK Tile programming model. Two variants are shown:
one with a regular implementation of the grouped GEMM operation and one with a persistent `TileLoop` kernel.

### Application flow

The application flow is the same for both examples; they only differ in the kernel instantation parameters.

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for the input matrices and the output matrix are created on the host.
3. The input matrices are initialized with random values.
4. Buffers for the input matrices and the output matrix are created on the device.
5. The input matrices are copied to the device.
6. CK Tile's `GroupedGemmKernel` is instantiated and launched on the device.
7. If validation is enabled, the results are compared against a CPU implementation using CK Tile's `reference_gemm`
   function.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of four key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this example it is set to CK Tile's
  `TileGemmShape`.
* A **problem** combines data types with the shape configuration. In this example the problems are set to CK Tile's
  `UniversalGemmPipelineProblem`.
* A **pipeline** schedules the sequence of operations for a kernel, such as the data loading, computation, and
  storage phases. In this example the pipelines are set to `GemmPipelineAgBgCrCompV3`.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `GroupedGemmKernel` is used.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::BaseGemmPipelineAgBgCrCompV3`
* `ck_tile::BaseGemmPipelineAgBgCrCompV4`
* `ck_tile::BaseGemmPipelineAgBgCrMem`
* `ck_tile::bool_constant`
* `ck_tile::CShuffleEpilogue`
* `ck_tile::CShuffleEpilogueProblem`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::GemmHostArgs`
* `ck_tile::GemmKernel`
* `ck_tile::GemmPipelineAgBgCrCompV3`
* `ck_tile::GemmPipelineAgBgCrCompV4`
* `ck_tile::GemmPipelineAgBgCrMem`
* `ck_tile::GemmPipelineAGmemBGmemCRegV1`
* `ck_tile::GemmPipelineProblem`
* `ck_tile::GemmPipelineScheduler`
* `ck_tile::GemmSpatiallyLocalTilePartitioner`
* `ck_tile::GemmTile1DPartitioner`
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

* `ck_tile::cast_pointer_to_constant_address_space`
* `ck_tile::check_err`
* `ck_tile::get_absolute_threshold`
* `ck_tile::get_relative_threshold`
* `ck_tile::host_tensor_descriptor`
* `ck_tile::integer_divide_ceil`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::reference_gemm`

### HIP runtime

#### Host symbols

* `hipFree`
* `hipMalloc`
* `hipMemcpy`
* `hipMemcpyWithStream`
