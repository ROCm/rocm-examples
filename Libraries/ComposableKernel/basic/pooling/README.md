# CK Tile Programming Model: Pooling Operator example

## Description

This example demonstrates how to use CK Tile's 3D pooling operator. Currently the pooling kernel only supports 2D and 3D
pooling. The pooling operation used in this example is `max`: for each output element, the maximum value in the
corresponding sampling window is returned.


### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for the input and output tensors are created on the host.
3. The input tensor is initialized with random floating-point values in the range $[-5, 5]$.
4. Buffers for the input and output tensors are created on the device.
5. The input tensor is copied from the host to the device.
6. CK Tile's built-in pool kernel is instantiated and launched on the device.
7. If validation is enabled the results are compared agains CK Tile's built-in `reference_pool3d` function.
8. All buffers are automatically freed.

## Key APIs and concepts

### CK Tile architecture

The example makes use of three key architectural components:

* **Shape** defines the hierarchical tile structure and memory layout. In this example, it is set to a 
`PoolShape` which is part of CK Tile.
* **Problem** combines data types with the shape configuration using CK Tile's `PoolProblem`.
* **Kernel** implements the actual computation using the problem definition. The example implementation uses CK Tile's
  `PoolKernel` kernel.

### Tile programming model

Internally, the kernel performs a **tilewise reduction**: Each thread block reduces a tile (block) of the input, using
shared memory and register accumulation for efficiency.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::PoolHostArgs`
* `ck_tile::PoolKernel`
* `ck_tile::PoolProblem`
* `ck_tile::PoolShape`
* `ck_tile::ReduceOp::Max`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::reference_pool3d`