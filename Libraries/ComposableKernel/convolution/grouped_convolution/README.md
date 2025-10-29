# CK Tile Programming Model: Grouped Convolution example

## Description

This example demonstrates how to perform a grouped convolution operation (forward pass) with the CK Tile programming
model.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for the input matrices and the output matrix are created on the host.
3. The input matrices are initialized with random values.
4. Buffers for the input matrices and the output matrix are created on the device.
5. The input matrices are copied to the device, the output matrix is initialized to `0` on the device.
6. CK Tile's `GroupedConvolutionForwardKernel` is instantiated and launched on the device.
7. If validation is enabled, the results are compared against CK Tile's `reference_grouped_conv_fwd` function.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of four key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this example it is set to CK Tile's
  `TileGemmShape`.
* A **problem** combines data types with the shape configuration. In this example the problem is set to CK Tile's
  `GemmPipelineProblem`.
* A **pipeline** schedules the sequence of operations for a kernel, such as the data loading, computation, and
  storage phases. In this example the pipeline is set to `GemmPipelineAGmemBGmemCRegV1`.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `GroupedConvolutionForwardKernel` is used.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::conv::ConvParam`
* `ck_tile::ConvolutionSpecialization`
* `ck_tile::CShuffleEpilogue`
* `ck_tile::CShuffleEpilogueProblem`
* `ck_tile::DeviceMem`
* `ck_tile::FillMonotonicSeq`
* `ck_tile::FillUniformDistribution`
* `ck_tile::GemmPipelineAGmemBGmemCRegV1`
* `ck_tile::GemmPipelineProblem`
* `ck_tile::GemmTile1DPartitioner`
* `ck_tile::GroupedConvHostArgs`
* `ck_tile::GroupedConvolutionForwardKernel`
* `ck_tile::GroupedConvTraits`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::integral_constant`
* `ck_tile::memory_operation_enum`
* `ck_tile::sequence`
* `ck_tile::stream_config`
* `ck_tile::tensor_layout::convolution::GKXC`
* `ck_tile::tensor_layout::convolution::GKYXC`
* `ck_tile::tensor_layout::convolution::GKZYXC`
* `ck_tile::tensor_layout::convolution::NDHWGC`
* `ck_tile::tensor_layout::convolution::NDHWGK`
* `ck_tile::tensor_layout::convolution::NHWGC`
* `ck_tile::tensor_layout::convolution::NHWGK`
* `ck_tile::tensor_layout::convolution::NWGC`
* `ck_tile::tensor_layout::convolution::NWGK`
* `ck_tile::TileGemmShape`
* `ck_tile::UniversalFlatmmPipelineAgBgCrPolicy`

#### Functions

* `ck_tile::check_err`
* `ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed`
* `ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed`
* `ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed`
* `ck_tile::get_absolute_threshold`
* `ck_tile::get_relative_threshold`
* `ck_tile::host_tensor_descriptor`
* `ck_tile::integer_divide_ceil`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::reference_grouped_conv_fwd`

### HIP runtime

#### Host symbols

* `dim3`
