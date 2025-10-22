# CK Tile Programming Model: Copy Kernel example

## Description

This example demonstrates how to perform memory copy operations using CK Tile. It introduces the fundamental building
blocks of CK Tile and provides a sandbox for experimenting with kernel parameters. The calculation performed by this
example is a simple copy operation: $\mathbf{Y} = \mathbf{X}$, where $\mathbf{X}$ is the input tensor
and $\mathbf{Y}$ is the output tensor.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for $\mathbf{X}$ and $\mathbf{Y}$ are created on the host.
3. $\mathbf{X}$ is initialized with increasing values $(1, 2, 3, \ldots)$.
4. Buffers for $\mathbf{X}$ and $\mathbf{Y}$ are created on the device.
5. $\mathbf{X}$ is copied from the host to the device.
6. The copy kernel is instantiated and launched on the device.
7. If validation is enabled, results are verified against the $\mathbf{X}$.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of three key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this example it is set to a custom 
  `TileCopyShape`.
* A **problem** combines data types with the shape configuration. In this example it is set to a custom 
  `TileCopyProblem`.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example a custom
  `TileCopyKernel` is used.

### Hierarchical tile structure

* A **vector**  specifies the number of contiguous elements processed by a single thread, enabling vectorized memory
  operations.
* A **wave tile** defines elements covered by a single wavefront.
* A **block tile** specifies elements covered by one block (typically mapped to one compute unit).
* **Block waves** define the number of concurrent waves active in a block.

### Tensor operations

* Tensor views are created using `make_naive_tensor_view` to combine memory buffers with tensor descriptors.
* Tile windows are created using `make_tile_window` to provide distributed access to tensor tiles.
* Data is loaded from global memory to registers using `load_tile`.
* Data is stored from registers to global memory using `store_tile`.
* Tile windows are moved to process subsequent tiles using `move_tile_window`.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::DeviceMem`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::number`
* `ck_tile::sequence`
* `ck_tile::stream_config`
* `ck_tile::tile_distribution_encoding`
* `ck_tile::tuple`

#### Functions

* `ck_tile::check_err`
* `ck_tile::get_block_id`
* `ck_tile::get_warp_id`
* `ck_tile::get_warp_size`
* `ck_tile::integer_divide_ceil`
* `ck_tile::launch_kernel`
* `ck_tile::load_tile`
* `ck_tile::make_kernel`
* `ck_tile::make_merge_transform`
* `ck_tile::make_naive_tensor_view`
* `ck_tile::make_pass_through_transform`
* `ck_tile::make_static_distributed_tensor`
* `ck_tile::make_static_tile_distribution`
* `ck_tile::make_tile_window`
* `ck_tile::move_tile_window`
* `ck_tile::store_tile`

### HIP runtime

#### Device symbols

* `__builtin_amdgcn_readfirstlane`
* `__syncthreads`
