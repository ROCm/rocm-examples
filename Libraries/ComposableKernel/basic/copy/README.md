# CK Tile Programming Model: Copy Kernel example

## Description

This example demonstrates how to perform memory copy operations using CK Tile. It introduces the fundamental building
blocks of CK Tile and provides a sandbox for experimenting with kernel parameters. The calculation performed by this
example is a simple copy operation: $\mathbf{Y} = \mathbf{X}$, where $\mathbf{X}$ is the input tensor
and $\mathbf{Y}$ is the output tensor.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Input tensor $\mathbf{X}$ is created on the host and copied to the device:
   1. A host buffer is allocated and initialized with increasing values (1, 2, 3, ...).
   2. The buffer is copied to the device using `DeviceMem::ToDevice`.
3. Output tensor $\mathbf{Y}$ is created on the device:
   1. A device buffer is allocated using `DeviceMem`.
4. The CK Tile kernel is configured with appropriate shape, problem, and policy definitions.
5. The copy kernel is launched on the device using `launch_kernel`.
6. $\mathbf{Y}$ is copied back to the host using `DeviceMem::FromDevice`.
7. If validation is enabled, results are verified against the input tensor using exact equality.
8. Performance metrics are collected over multiple iterations.
9. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

CK Tile is built around four key architectural components:

* **Shape** defines the hierarchical tile structure and memory layout using `TileCopyShape`.
* **Problem** combines data types with the shape configuration using `TileCopyProblem`.
* **Policy** defines memory access patterns and distribution strategies using `TileCopyPolicy`.
* **Kernel** implements the actual computation using the problem and policy definitions.

### Hierarchical tile structure

* **ThreadTile**  specifies the number of contiguous elements processed by a single thread, enabling vectorized memory
  operations.
* **WaveTile** defines elements covered by a single wave.
* **BlockTile** specifies elements covered by one block (typically mapped to one CU).
* **BlockWaves** determines the number of concurrent waves active in a block.

### Tensor operations

* Tensor views are created using `make_naive_tensor_view` to combine memory buffers with tensor descriptors.
* Tile windows are created using `make_tile_window` to provide distributed access to tensor tiles.
* Data is loaded from global memory to registers using `load_tile`.
* Data is stored from registers to global memory using `store_tile`.
* Tile windows are moved to process subsequent tiles using `move_tile_window`.

### Available kernel implementations

The example provides three different kernel implementations:

* `TileCopyKernel`: Direct copy from global memory to global memory.
* `ElementWiseTileCopyKernel`: Element-wise copy allowing for data transformation during the copy process.
* `TileCopyKernel_LDS`: Copy through LDS (Local Data Share) memory for scenarios requiring data staging.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::DeviceMem`
* `ck_tile::ElementWiseTileCopyKernel`
* `ck_tile::HostTensor`
* `ck_tile::sequence`
* `ck_tile::TileCopyShape`
* `ck_tile::TileCopyProblem`
* `ck_tile::TileCopyPolicy`
* `ck_tile::TileCopyKernel`
* `ck_tile::TileCopyKernel_LDS`
* `ck_tile::tile_distribution_encoding`
* `ck_tile::tuple`

#### Functions

* `ck_tile::block_sync_lds`
* `ck_tile::check_err`
* `ck_tile::get_block_id`
* `ck_tile::get_warp_size`
* `ck_tile::integer_divide_ceil`
* `ck_tile::launch_kernel`
* `ck_tile::load_tile`
* `ck_tile::make_kernel`
* `ck_tile::make_naive_tensor_view`
* `ck_tile::make_static_distributed_tensor`
* `ck_tile::make_static_tile_distribution`
* `ck_tile::make_tile_window`
* `ck_tile::move_tile_window`
* `ck_tile::store_tile`
* `ck_tile::sweep_tile_span`

### HIP runtime

* `__builtin_amdgcn_readfirstlane`
