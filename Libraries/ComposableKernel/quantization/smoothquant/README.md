# CK Tile Programming Model: SmoothQuant example

## Description

This example shows how to implement SmoothQuant using the CK Tile programming model. There are two variants: One basic
example with a fixed parameter set (`example_`) and one example with various available parameters.

### Supported architectures

The example is supported for the following GPU architectures:

* `gfx908`
* `gfx90a`
* `gfx942`
* `gfx950`

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers are created for all tensors on the host.
3. The input tensors are initialized with random values.
4. Buffers are created for all tensors on the device.
5. The input tensors are copied to the device.
6. CK Tile's `Smoothquant` kernel is instantiated and launched on the device.
7. If validation is enabled, the results are compared against an implementation using CK Tile's `reference_` functions.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of four key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this example it is set to CK Tile's
  `Generic2dBlockShape`.
* A **problem** combines data types with the shape configuration. In this example the problem is set to CK Tile's
  `SmoothquantPipelineProblem`.
* A **pipeline** schedules the sequence of operations for a kernel, such as the data loading, computation, and
  storage phases. In this example the pipeline is set to either `SmoothquantPipelineOnePass` or
  `SmoothquantPipelineTwoPass`, depending on the chosen parameters.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `Smoothquant` is used.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::Generic2dBlockShape`
* `ck_tile::fp16_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::Smoothquant`
* `ck_tile::SmoothquantHostArgs`
* `ck_tile::remove_cvref_t`
* `ck_tile::sequence`
* `ck_tile::SmoothquantPipelineOnePass`
* `ck_tile::SmoothquantPipelineProblem`
* `ck_tile::SmoothquantPipelineTwoPass`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_ParallelTensorFunctor`
* `ck_tile::make_tuple`
* `ck_tile::reference_reduce`
* `ck_tile::reference_rowwise_quantization2d`
* `ck_tile::reference_unary_elementwise`
* `ck_tile::type_convert`

### HIP runtime

#### Types

* `dim3`
