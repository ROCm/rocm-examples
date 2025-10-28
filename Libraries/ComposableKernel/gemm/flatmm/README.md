# CK Tile Programming Model: FLATMM example

## Description

This example demonstrates how to perform a FLATMM operation with the CK Tile programming model.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for the input matrices and the output matrix are created on the host.
3. The input matrices are initialized with random values.
4. Buffers for the input matrices and the output matrix are created on the device.
5. The input matrices are copied to the device, the output matrix is initialized to `0` on the device.
6. CK Tile's `FlatmmKernel` is instantiated and launched on the device.
7. If validation is enabled, the results are compared against CK Tile's `reference_gemm` function or
   `reference_gemm_gpu` kernel.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of four key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this example it is set to CK Tile's
  `TileFlatmmShape`.
* A **problem** combines data types with the shape configuration. In this example the problem is set to CK Tile's
  `GemmPipelineProblem`.
* A **pipeline** schedules the sequence of operations for a kernel, such as the data loading, computation, and
  storage phases. In this example the pipeline is set to `FlatmmPipelineAGmemBGmemCRegV1`.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `FlatmmKernel` is used.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::bf8_t`
* `ck_tile::bool_constant`
* `ck_tile::CShuffleEpilogue`
* `ck_tile::CShuffleEpilogueProblem`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::FlatmmHostArgs`
* `ck_tile::FlatmmPipelineAGmemBGmemCRegV1`
* `ck_tile::fp8_t`
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
* `ck_tile::TileFlatmmShape`
* `ck_tile::TileGemmTraits`
* `ck_tile::UniversalFlatmmPipelineAgBgCrPolicy`

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
