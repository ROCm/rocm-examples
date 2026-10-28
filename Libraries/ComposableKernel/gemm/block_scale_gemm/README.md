# CK Tile Programming Model: Block-Scale GEMM example

## Description

This example demonstrates how to perform a block-scale GEMM operation with the
CK Tile programming model.

### Supported architectures

This example works with the following architectures:

* `gfx942`
* `gfx950`

### Application flow

1. Command line arguments are parsed to configure matrix dimensions, data
   type (`-prec`), quantization mode (`-quant_mode`) and quantization group
   size (`-group_size`).
2. The arguments are hashed into a key that selects a kernel from a lookup
   table. Each `gemm_bquant_quantgrouped_<prec>.cpp` file registers its kernel
   instances in this table during static initialization.
3. Buffers for the input matrices, the B scale tensor and the output matrix are
   created on the host.
4. The input matrices and the scale tensor are initialized with random values.
5. Buffers for the input matrices, the scale tensor and the output matrix are
   created on the device.
6. The inputs are copied to the device, the output matrix is initialized to `0`
   on the device.
7. CK Tile's `QuantGemmKernel` is instantiated with the
   `BQuantGemmPipelineAgBgCrCompV3` pipeline and launched on the device.
8. If validation is enabled, the results are compared against CK Tile's
   `reference_gemm_quant` function.
9. All buffers are freed automatically.

This example registers BQuant kernels without preshuffling for the `fp8`,
`bf8`, `fp8i4` and `bf8i4` data types. In BQuant mode, each
`GroupSize` × `N` block of the B matrix shares one scale, giving a B scale
tensor of shape `[K/GroupSize, N]`. The A matrix is not quantized.

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

* `ck_tile::ABQuantGemmPipelineAgBgCrCompV3`
* `ck_tile::ABQuantGemmPipelineAgBgCrEightWaves`
* `ck_tile::AQuantGemmPipelineAgBgCrCompV3`
* `ck_tile::AQuantGemmPipelineAgBgCrMem`
* `ck_tile::ArgParser`
* `ck_tile::BaseGemmPipelineAgBgCrCompV3`
* `ck_tile::BaseGemmPipelineAgBgCrMem`
* `ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2`
* `ck_tile::bf16_t`
* `ck_tile::bf8_t`
* `ck_tile::bool_constant`
* `ck_tile::BQuantGemmPipelineAgBgCrCompV3`
* `ck_tile::CastPolicy`
* `ck_tile::CShuffleEpilogue`
* `ck_tile::CShuffleEpilogueProblem`
* `ck_tile::DataTypeTraits`
* `ck_tile::DeviceMem`
* `ck_tile::e8m0_t`
* `ck_tile::element_wise::PassThrough`
* `ck_tile::FillConstant`
* `ck_tile::FillUniformDistribution`
* `ck_tile::FillUniformDistributionIntegerValue`
* `ck_tile::fp8_t`
* `ck_tile::GemmABQuantPipelineProblem`
* `ck_tile::GemmAQuantPipelineProblem`
* `ck_tile::GemmBQuantPipelineProblem`
* `ck_tile::GemmPipelineAgBgCrCompV3`
* `ck_tile::GemmPipelineProblemBase`
* `ck_tile::GemmPipelineScheduler`
* `ck_tile::GemmRowColTensorQuantPipelineProblem`
* `ck_tile::GemmTile1DPartitioner`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::kernel_attr`
* `ck_tile::MicroscaleGemmPipelineAgBgCrCompV3`
* `ck_tile::number`
* `ck_tile::PermuteNEpilogue`
* `ck_tile::PermuteNEpilogueProblem`
* `ck_tile::pk_fp4_t`
* `ck_tile::pk_int4_t`
* `ck_tile::QuantGemmHostArgs`
* `ck_tile::QuantGemmKernel`
* `ck_tile::QuantGroupShape`
* `ck_tile::QuantType`
* `ck_tile::remove_cvref_t`
* `ck_tile::RotatingMemWrapper`
* `ck_tile::sequence`
* `ck_tile::stream_config`
* `ck_tile::TailNumber`
* `ck_tile::tensor_layout::gemm::ColumnMajor`
* `ck_tile::tensor_layout::gemm::RowMajor`
* `ck_tile::TileGemmQuantTraits`
* `ck_tile::TileGemmShape`
* `ck_tile::tuple`
* `ck_tile::WPABQuantBPipelineAgBgCrV2`
* `ck_tile::WPQuantBPipelineAgBgCrV2`

#### Functions

* `ck_tile::bq_permuteN`
* `ck_tile::check_err`
* `ck_tile::flush_icache`
* `ck_tile::get_absolute_threshold`
* `ck_tile::get_default_stride`
* `ck_tile::get_k_warp_tile`
* `ck_tile::get_relative_threshold`
* `ck_tile::get_splitk_batch_k_read`
* `ck_tile::hip_check_error`
* `ck_tile::host_tensor_descriptor`
* `ck_tile::integer_divide_ceil`
* `ck_tile::integer_least_multiple`
* `ck_tile::launch_kernel`
* `ck_tile::launch_kernel_time_mask`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::permute_vectors_i4x4_b`
* `ck_tile::reference_gemm_abquant`
* `ck_tile::reference_gemm_quant`
* `ck_tile::reference_gemm_rowcol_quant`
* `ck_tile::reference_gemm_tensor_quant`
* `ck_tile::reference_mx_gemm_bquant`
* `ck_tile::shuffle_aq`
* `ck_tile::shuffle_b`
* `ck_tile::shuffle_b_permuteN`
* `ck_tile::shuffle_bq`
