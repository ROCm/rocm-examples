# CK Tile Programming Model: Reduction example

## Description

This example demonstrates parallel reduction (sum, max, etc.) using the CK Tile programming model, a core operation for
normalization, statistics, and aggregation in deep learning.

Given a tensor $\mathbf{X}$ and a reduction axis, the performed computation is $\mathbf{Y} = \sum_i \mathbf{X}_i$.

### Supported architectures

This example works with
[all supported GPU architectures](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. $\mathbf{X}$ is created on the host and initialized with random floating-point values in the range $[-5, 5]$.
3. Buffers for $\mathbf{X}$ and $\mathbf{Y}$ are created on the device.
4. $\mathbf{X}$ is copied from the host to the device.
5. A custom reduction kernel is instantiated and launched on the device.
6. If validation is enabled the results are compared agains CK Tile's built-in `reference_reduce` function.
7. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The CK Tile framework is built around four key architectural components:

The CK Tile framework is built around four key architectural components:

* The **shape** defines the hierarchical tile structure and memory layout.
* The **problem** combines data types with the shape configuration.
* The **pipeline** schedules the sequence of operations for a kernel.
* The **kernel** implements the actual computation using the problem and pipeline definitions.

For more information on CK Tile terminology, refer to the
[Composable Kernel Glossary](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/reference/Composable-Kernel-Glossary.html).

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
* `ck_tile::ReduceOp::Add`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::block_tile_reduce`
* `ck_tile::block_tile_reduce_sync`
* `ck_tile::cast_tile`
* `ck_tile::check_err`
* `ck_tile::get_block_id`
* `ck_tile::integer_divide_ceil`
* `ck_tile::launch_kernel`
* `ck_tile::load_tile`
* `ck_tile::make_kernel`
* `ck_tile::make_naive_tensor_view`
* `ck_tile::make_naive_tensor_view_packed`
* `ck_tile::make_tile_window`
* `ck_tile::make_tuple`
* `ck_tile::move_tile_window`
* `ck_tile::reduce_on_sequence`
* `ck_tile::reference_reduce`
* `ck_tile::set_tile`
* `ck_tile::store_tile`

### HIP Runtime

#### Device symbols

* `__builtin_amdgcn_readfirstlane`
