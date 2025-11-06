# CK Tile Programming Model: Elementwise example

## Description

This example demonstrates how to perform various elementwise operations on
tensors using the CK Tile programming model:

* An addition of two matrices (`elementwise_example.cpp`)
* A square of a single tensor (`elementwise_example_unary.cpp`)
* A transpose of a single tensor (`elementwise_example_transpose.cpp`)
* An addition of two 4D tensors (`elementwise_example_add_4d.cpp`)

### Supported architectures

The example is available for
[all supported GPU architectures](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

### Application flow

The application flow is the same for all examples; they only differ in their
parameters for the kernel instantiation.

1. Command line arguments are parsed to configure matrix dimensions and
   execution parameters.
2. Host buffers are created for the input and output tensors.
3. The input tensor(s) is (are) initialized with random values.
4. Device buffers are created for the input and output tensors.
5. The input tensor(s) is (are) copied from the host to the device.
6. The problem is defined.
7. The kernel is instantiated and launched on the device.
8. If enabled via command-line parameter, the results are compared against
   CK Tile's `reference_{binary,transpose,unary}_elementwise` implementations.
9. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of three key architectural components:

* **Shape** defines the hierarchical tile structure and memory layout. In this
  application, it is set to CK Tile's `ElementWiseShape` which is part of
  CK Tile.
* **Problem** combines data types with the shape configuration using CK Tile's
  `ElementWisePipelineProblem`.
* **Kernel** implements the actual computation using the problem definition.
  The example implementation uses CK Tile's `ElementWiseKernel`.

### Tile Programming Model

Internally, each thread block processes a **tile** (block of patches). The *problem* is defined as a modular
**pipeline** which can be extended for fused operations (e.g., quantization, activation).

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::DeviceMem`
* `ck_tile::ElementWiseDefaultPolicy`
* `ck_tile::ElementWiseKernel`
* `ck_tile::ElementWisePipelineProblem`
* `ck_tile::ElementWiseShape`
* `ck_tile::element_wise::Add`
* `ck_tile::element_wise::PassThrough`
* `ck_tile::element_wise::UnarySquare`
* `ck_tile::FillUniformDistribution`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::number`
* `ck_tile::sequence`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::get_warp_size`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::reference_binary_elementwise`
* `ck_tile::reference_transpose_elementwise`
* `ck_tile::reference_unary_elementwise`
