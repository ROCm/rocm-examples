# CK Tile Programming Model: Layernorm2D example

## Description

This example demonstrates how to perform the forward pass of the Layernorm2D operation with the CK Tile programming
model.

### Supported architectures

The example is supported for the following GPU architectures:

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
6. CK Tile's built-in `Layernorm2dFwd` kernel is instantiated and launched on the device.
7. If validation is enabled, the results are compared against CK Tile's `reference_layernorm2d_fwd` function.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of four key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this example it is set to CK Tile's
  `Generic2dBlockShape`.
* A **problem** combines data types with the shape configuration. In this example it is set to CK Tile's
  `Layernorm2dFwdPipelineProblem`.
* A **pipeline** schedules the sequence of operations for a kernel, such as the data loading, computation, and
  storage phases. In this example it is set to CK Tile's `Layernorm2dFwdPipelineOnePass` or
  `Layernorm2dFwdPipelineTwoPass`, depending on the chosen parameters.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `Layernorm2dFwd` kernel is used.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
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
* `ck_tile::Layernorm2dFusedAddEnum`
* `ck_tile::Layernorm2dFusedQuantEnum`
* `ck_tile::Layernorm2dFwd`
* `ck_tile::Layernorm2dFwdHostArgs`
* `ck_tile::Layernorm2dFwdPipelineOnePass`
* `ck_tile::Layernorm2dFwdPipelineProblem`
* `ck_tile::Layernorm2dFwdPipelineTwoPass`
* `ck_tile::Layernorm2dFwdTraits`
* `ck_tile::Layernorm2dXBiasEnum`
* `ck_tile::null_type`
* `ck_tile::remove_cvref_t`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::reference_layernorm2d_fwd`
* `ck_tile::type_convert`
