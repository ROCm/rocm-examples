# CK Tile Programming Model: GEMM example

## Description

This example demonstrates how to perform a GEMM operation using CK Tile. Five
variants are shown:

* **Basic**: Basic GEMM pipeline (`gemm_basic.cpp`)
* **Universal**: Memory bound pipeline optimized for various problem sizes (`universal_gemm.cpp`)
* **Weight Preshuffle**: Includes extra weighting and preshuffling steps (`gemm_weight_preshuffle.cpp`)
* **Split-K Reduce**: Reduction kernel for split-K GEMM (`gemm_splitk_two_stage_reduce.cpp`)
* **Split-K Two-Stage**: Two-stage split-K GEMM implementation (`gemm_splitk_two_stage.cpp`)

### Supported architectures

The example is supported for the following architectures:

* `gfx908`
* `gfx90a`
* `gfx942`
* `gfx950`

### Application flow

The application flow is the same for all examples; they only differ in the
kernel instantation parameters.

1. Command line arguments are parsed to configure matrix dimensions and
   execution parameters.
2. Buffers for the input matrices and the output matrix are created on the host.
3. The input matrices are initialized with random values.
4. Buffers for the input matrices and the output matrix are created on the
   device.
5. The input matrices are copied to the device.
6. CK Tile's `GemmKernel` is instantiated and launched on the device.
7. If validation is enabled, the results are compared against CK Tile's
   `reference_gemm` function or `reference_gemm_gpu` kernel.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of four key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this
  example it is set to CK Tile's `TileGemmShape`.
* A **problem** combines data types with the shape configuration. In this
  example the problems are set to CK Tile's `GemmPipelineProblem` (basic
  example) and `UniversalGemmPipelineProblem` (other examples).
* A **pipeline** schedules the sequence of operations for a kernel, such as the
  data loading, computation, and storage phases. In this example the pipelines
  are set to `GemmPipelineAGmemBGmemCRegV1` (basic example),
  `GemmPipelineAgBgCrCompV3` (universal example), and
  `WeightPreshufflePipelineAGmemBGmemCRegV2` (`weight_preshuffle` example).
* A **kernel** implements the actual computation using the problem and policy
  definitions. In this example CK Tile's `GemmKernel` is used.

## Used API surface

### CK Tile

#### Types

* `ck_tile::AdjustToStructuredSparsity`
* `ck_tile::ArgParser`
* `ck_tile::BaseGemmPipelineAgBgCrCompV3`
* `ck_tile::BaseGemmPipelineAgBgCrCompV4`
* `ck_tile::BaseGemmPipelineAgBgCrMem`
* `ck_tile::bf16_t`
* `ck_tile::bf8_t`
* `ck_tile::bool_constant`
* `ck_tile::CShuffleEpilogue`
* `ck_tile::CShuffleEpilogueProblem`
* `ck_tile::DeviceMem`
* `ck_tile::FillMonotonicSeq`
* `ck_tile::FillUniformDistribution`
* `ck_tile::fp8_t`
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
* `ck_tile::pk_int4_t`
* `ck_tile::remove_cvref_t`
* `ck_tile::sequence`
* `ck_tile::stream_config`
* `ck_tile::TailNumber`
* `ck_tile::tensor_layout::gemm::ColumnMajor`
* `ck_tile::tensor_layout::gemm::RowMajor`
* `ck_tile::TileGemmShape`
* `ck_tile::TileGemmTraits`
* `ck_tile::TileGemmUniversalTraits`
* `ck_tile::UniversalGemmPipelineProblem`
* `ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2`

#### Functions

* `ck_tile::check_err`
* `ck_tile::get_absolute_threshold`
* `ck_tile::get_relative_threshold`
* `ck_tile::host_tensor_descriptor`
* `ck_tile::integer_divide_ceil`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::reference_gemm`
* `ck_tile::reference_gemm_gpu`

### HIP runtime

#### Host symbols

* `hipFree`
* `hipMalloc`
* `hipMemcpy`
