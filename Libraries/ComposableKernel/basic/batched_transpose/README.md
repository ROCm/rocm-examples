# CK Tile Programming Model: Batched Transpose example

## Description

This example demonstrates batched tensor transpose CK Tile. It supports common
layout conversions such as NCHW to NHWC and NHWC to NCHW, which are essential
for deep learning frameworks and hardware accelerators.

Currently, the example supports batched transpose operations in two directions:

* NCHW to NHWC
* NHWC to NCHW

This enables two transpose patterns from NCHW: NCHW to NHWC, or NCHW to NWCH.
The implementation supports multiple data types including `fp8`, `fp16`, and
`bf16`. On `gfx950` architecture, an LDS-accelerated pipeline mode is available
for improved performance. The current implementation performs transpose
operations with single data point reads. Vectorized transpose support will be
added in a future release.

The example performs the following computation:

Given a batch of tensors $\mathbf{X}$ of shape $[N, C, H, W]$, the transpose
operation rearranges axes to produce $\mathbf{Y}$ of shape $[N, H, W, C]$
(NCHW to NHWC) or other permutations. For each element:

$$
Y_{n, h, w, c} = X_{n, c, h, w}
$$

### Supported architectures

The example is supported for the following GPU architectures:

* `gfx908`
* `gfx90a`
* `gfx942`
* `gfx950`

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for $\textbf{X}$ and $\textbf{Y}$ are created on the host.
3. $\textbf{X}$ is initialized with random floating-point values in the range $[-0.5, 0.5]$.
4. Buffers for $\textbf{X}$ and $\textbf{Y}$ are created on the device.
5. $\textbf{X}$ is copied from the host to the device.
6. CK Tile's built-in batched transpose kernel is instantiated and launched on the device.
7. $\textbf{Y}$ is copied from the device to the host.
8. If verification is enabled, the results are compared against CK Tile's `reference_batched_transpose` function.
9. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of three key architectural components:

* A **problem** combines data types using CK Tile's `BatchedTransposeProblem`.
* A **pipeline** schedules the sequence of operations for a kernel, such as the data loading, computation, and
  storage phases. In this example it is set to CK Tile's `BatchedTransposePipeline`.
* A **kernel** implements the actual computation using the problem definition. The example implementation uses
  CK Tile's `BatchedTransposeKernel` kernel.

### Tile programming model

Internally, the kernel performs a **tilewise batched transpose**: each thread block processes a tile (block) of the
input, computes the permuted indices, and writes to the output.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::BatchedTransposeHostArgs`
* `ck_tile::BatchedTransposeKernel`
* `ck_tile::BatchedTransposePipeline`
* `ck_tile::BatchedTransposeProblem`
* `ck_tile::bf16_t`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::fp16_t`
* `ck_tile::fp8_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::reference_batched_transpose`
* `ck_tile::type_convert`
