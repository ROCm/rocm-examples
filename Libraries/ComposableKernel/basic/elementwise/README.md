# CK Tile Programming Model: Element-wise example

## Description

This example demonstrates how to perform element-wise operations (add, unary operations, transpose) using the CK Tile
programming model.

Given an input tensor $\mathbf{X}$, the shown computations are:

* Element-wise add: $\mathbf{Y} = \mathbf{X}_a + \mathbf{X}_b$. `elementwise_example.cpp` shows how to perform the
  operation for a 2D tensor while `elementwise_example_add_4d.cpp` demonstrates how to add 4D tensors element-wise.
* Unary operations: $\mathbf{Y} = \text{op}(\mathbf{X})$, where $\text{op}$ is either an element-wise square function
  or an element-wise type conversion.
* Transpose: $\mathbf{Y} = \mathbf{X}^{\text{T}}$.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for $\mathbf{X}$ and $\mathbf{Y}$ are created on the host.
3. $\mathbf{X}$ is initialized with random floating-point values in the range $[0, 5]$.
3. Buffers for $\mathbf{X}$ and $\mathbf{Y}$ are created on the device.
4. $\mathbf{X}$ is copied from the host to the device.
5. CK Tile's built-in element-wise kernel is instantiated and launched on the device.
6. If validation is enabled the results are compared agains CK Tile's built-in
   `reference_{binary,transpose,unary}_elementwise` functions.
7. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of four key architectural components:

* **Shape** defines the hierarchical tile structure and memory layout. In this example, it is set to a 
`ElementWiseShape` which is part of CK Tile.
* **Problem** combines data types with the shape configuration using CK Tile's `ElementWisePipelineProblem`.
* **Policy** defines memory access patterns and distribution strategies. In this example it is set to CK Tile's
  `ElementWiseDefaultPolicy`.
* **Kernel** implements the actual computation using the problem definition. The example implementation uses CK Tile's
  `ElementWiseKernel` kernel.

### Tile programming model

Internally, the kernel performs a **tilewise operation**: Each thread block computes a tile (block) of the input, using
shared memory and register accumulation for efficiency.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::DeviceMem`
* `ck_tile::ElementWiseDefaultPolicy`
* `ck_tile::ElementWiseKernel`
* `ck_tile::ElementWisePipelineProblem`
* `ck_tile::element_wise::Add`
* `ck_tile::FillUniformDistribution`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::number`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::dump_elementwise_json_results`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::reference_binary_elementwise`
* `ck_tile::reference_transpose_elementwise`
* `ck_tile::reference_unary_elementwise`