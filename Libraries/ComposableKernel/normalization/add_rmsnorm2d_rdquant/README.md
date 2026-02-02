# CK Tile Programming Model: Add + RMSNorm2D + Row-wise Dynamic Quantization example

## Description

This example demonstrates how to perform add + RMSNorm2D + row-wise dynamic quantization forward operations using the
CK Tile programming model. RDQuant is short for row-wise dynamic quantization. There are two variants: One basic
example with a fixed tile size ("_tile") and one with flexible tile size depending on the chosen parameters.

### Supported architectures

This example works with the following GPU architectures:

* `gfx908`
* `gfx90a`
* `gfx942`
* `gfx950`

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for all tensors are created on the host.
3. The input tensors are initialized with random values.
4. Buffers for all tensors are created on the device.
5. The input tensors are copied from the host to the device.
6. CK Tile's built-in `AddRmsnorm2dRdquantFwd` kernel is instantiated and launched on the device.
7. If validation is enabled, the results are compared against CK Tile's `reference_rmsnorm2d_fwd` function.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The CK Tile framework is built around four key architectural components:

* The **shape** defines the hierarchical tile structure and memory layout.
* The **problem** combines data types with the shape configuration.
* The **pipeline** schedules the sequence of operations for a kernel.
* The **kernel** implements the actual computation using the problem and pipeline definitions.

For more information on CK Tile terminology, refer to the
[Composable Kernel Glossary](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/reference/Composable-Kernel-Glossary.html).

## Used API surface

### CK Tile

#### Types

* `ck_tile::AddRmsnorm2dRdquantFwd`
* `ck_tile::AddRmsnorm2dRdquantFwdHostArgs`
* `ck_tile::AddRmsnorm2dRdquantFwdPipelineOnePass`
* `ck_tile::AddRmsnorm2dRdquantFwdPipelineProblem`
* `ck_tile::AddRmsnorm2dRdquantFwdPipelineThreePass`
* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::fp16_t`
* `ck_tile::fp8_t`
* `ck_tile::Generic2dBlockShape`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::int8_t`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::reference_binary_elementwise`
* `ck_tile::reference_reduce`
* `ck_tile::reference_rmsnorm2d_fwd`
* `ck_tile::reference_rowwise_quantization2d`
* `ck_tile::reference_unary_elementwise`
* `ck_tile::type_convert`
