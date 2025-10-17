# CK Tile Programming Model: Reduction example

## Description

This example demonstrates parallel reduction (sum, max, etc.) using the CK Tile programming model, a core operation for
normalization, statistics, and aggregation in deep learning.

Given a tensor $\mathbf{X}$ and a reduction axis, the performed computation is $\mathbf{Y} = \sum_i \mathbf{X}_i$.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. $\mathbf{X}$ is created on the host and initialized with random floating-point values in the range $[-5, 5]$.
3. Buffers for $\mathbf{X}$ and $\mathbf{Y}$ are created on the device.
4. $\mathbf{X}$ is copied from the host to the device.
5. CK Tile's built-in reduction kernel is instantiated and launched on the device.
6. If validation is enabled the results are compared agains CK Tile's built-in `reference_reduce` function.
7. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of three key architectural components:

* **Shape** defines the hierarchical tile structure and memory layout. In this example, it is set to a 
`Reduce2dShape` which is part of CK Tile.
* **Problem** combines data types with the shape configuration using CK Tile's `Reduce2dProblem`.
* **Kernel** implements the actual computation using the problem definition. The example implementation uses CK Tile's
  `Reduce` kernel.

### Tile programming model

Internally, the kernel performs a **tilewise reduction**: Each thread block reduces a tile (block) of the input, using
shared memory and register accumulation for efficiency.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::number`
* `ck_tile::Reduce`
* `ck_tile::Reduce2dProblem`
* `ck_tile::Reduce2dShape`
* `ck_tile::ReduceOp::Add`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::dump_reduce_json_results`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::reference_reduce`
