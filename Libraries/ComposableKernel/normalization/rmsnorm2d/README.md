# CK Tile Programming Model: RMSNorm2D example

## Description

This example demonstrates how to perform the forward pass of RMSNorm2D using the CK Tile programming model. There are
two variants: One basic example with a fixed parameter set (`example_`) and one with various different available
parameters.

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
6. CK Tile's built-in `Rmsnorm2dFwd` kernel is instantiated and launched on the device.
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

* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::Default2DAndDynamicQuantEpilogue`
* `ck_tile::Default2DAndDynamicQuantEpilogueProblem`
* `ck_tile::Default2DAndDynamicQuantEpilogueTraits`
* `ck_tile::Default2DEpilogue`
* `ck_tile::Default2DEpilogueProblem`
* `ck_tile::DeviceMem`
* `ck_tile::DynamicQuantEpilogue`
* `ck_tile::DynamicQuantEpilogueProblem`
* `ck_tile::DynamicQuantEpilogueTraits`
* `ck_tile::FillUniformDistribution`
* `ck_tile::fp8_t`
* `ck_tile::Generic2dBlockShape`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::int8_t`
* `ck_tile::Rmsnorm2dFwd`
* `ck_tile::Rmsnorm2dFwdHostArgs`
* `ck_tile::Rmsnorm2dFwdPipelineOnePass`
* `ck_tile::Rmsnorm2dFwdPipelineProblem`
* `ck_tile::Rmsnorm2dFwdPipelineTwoPass`
* `ck_tile::Rmsnorm2dFwdTraits`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::reference_rmsnorm2d_fwd`
* `ck_tile::type_convert`
