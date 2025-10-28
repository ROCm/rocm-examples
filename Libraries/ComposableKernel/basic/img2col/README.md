# CK Tile Programming Model: Image to Column (im2col) with CK Tile

## Description

This example demonstrates the $\text{im2col}$ transformation using the CK Tile programming model. This transformation is
a key step for converting convolution operations into general matrix multiplication (GEMM) for efficient GPU execution.

Given an input image tensor $\mathbf{X}$ and a convolution kernel size, `im2col` rearranges sliding windows of
$\mathbf{X}$ into columns:

* Each patch is flattened and stacked as a column in the output matrix.
* This enables convolution as matrix multiplication: $\text{im2col}(X) \times \mathbf{W}$.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. A descriptor for the input tensor is created.
3. Host buffers are created for the input and output tensors.
4. The input tensor is initialized with random values in the range of either $[-5, 5]$ (integer values) or
   $[-0.5, 0.5]$ (floating-point values), depending on the chosen command line parameter.
5. Device buffers are created for the input and output tensors.
6. The input tensor is copied from the host to the device.
7. The tensor shapes and the problem are defined.
8. The kernel is instantiated and launched on the device.
9. If enabled via command-line parameter, the kernel performance is measured.
10. If enabled via command-line parameter, the results are compared against CK Tile's `reference_im2col` implementation.
11. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of three key architectural components:

* **Shape** defines the hierarchical tile structure and memory layout. In this application, it is set to a
`TileImageToColumnShape` which is part of CK Tile.
* **Problem** combines data types with the shape configuration using CK Tile's `BlockImageToColumnProblem`.
* **Kernel** implements the actual computation using the problem definition. The example implementation uses CK Tile's
  `ImageToColumn` kernel.

### Tile Programming Model

Internally, each thread block processes a **tile** (block of patches). The *problem* is defined as a modular
**pipeline** which can be extended for fused operations (e.g., quantization, activation).

## Used API surface

### CK Tile

#### Types

* `ck_tile::BlockImageToColumnProblem`
* `ck_tile::conv::ConvParam`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::FillUniformDistributionIntegerValue`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::HostTensorDescriptor`
* `ck_tile::index_t`
* `ck_tile::ImageToColumn`
* `ck_tile::long_index_t`
* `ck_tile::sequence`
* `ck_tile::stream_config`
* `ck_tile::tensor_layout::convolution::NHWGC`
* `ck_tile::TileImageToColumnShape`

#### Functions

* `ck_tile::check_err`
* `ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::reference_im2col`
* `ck_tile::to_array`
