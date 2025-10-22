# CK Tile Programming Model: Fused Multi-Head Attention example

## Description

This example demonstrates how to perform FMHA (fused multi-head attention) with the CK Tile programming model. The files
with the `_fwd` suffix contain the algorithm's forward pass, while the `_bwd` files show the backward pass.

### Application flow

#### Forward pass

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for all tensors are created on the host and initialized with random values.
3. Buffers for all tensors are created on the device.
4. The tensors are copied from the host to the device.
5. CK Tile's built-in `fmha_fwd` kernel is instantiated and launched on the device.
6. If validation is enabled, the results are compared against CK Tile's naive implementation of the forward pass.
7. Additionally, the results are compared against a custom implementation using CK Tile's `reference_` functionality.
8. All buffers are freed automatically.

#### Backward pass

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for all tensors are created on the host and initialized with random values.
3. Buffers for all tensors are created on the device.
4. The tensors are copied from the host to the device.
5. CK Tile's built-in `fmha_bwd` kernel is instantiated and launched on the device.
6. If validation is enabled, the results are compared against a custom implementation using CK Tile's `reference_` 
   functionality.
7. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of three key architectural components:

* A **shape** defines the hierarchical tile structure and memory layout. In this example it is set to a custom 
  shape based on the sequence length.
* A **problem** combines data types with the shape configuration. In this example its components are set individually
  by specifying the individual tensors' dimensions, strides, etc. as part of the arguments to the kernel.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `fmha_fwd` and `fmha_bwd` are used.

## Used API surface

### CK Tile

#### Types

* `ck_tile::Alibi`
* `ck_tile::AlibiMode`
* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::bf8_t`
* `ck_tile::BlockAttentionBiasEnum`
* `ck_tile::BlockFmhaPipelineEnum`
* `ck_tile::composes`
* `ck_tile::DeviceMem`
* `ck_tile::FillNormalDistribution`
* `ck_tile::FillNormalDistributionIntegerValue`
* `ck_tile::FillTrigValue`
* `ck_tile::FillUniformDistribution`
* `ck_tile::FillUniformDistributionIntegerValue`
* `ck_tile::fp8_t`
* `ck_tile::GenericAttentionMask`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::identity`
* `ck_tile::index_t`
* `ck_tile::naive_attention_fwd_args`
* `ck_tile::naive_attention_fwd_traits`
* `ck_tile::number`
* `ck_tile::RotaryEmbeddingEnum`
* `ck_tile::remove_cvref_t`
* `ck_tile::saturates`
* `ck_tile::scales`
* `ck_tile::span`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::get_alibi_slopes`
* `ck_tile::integer_divide_ceil`
* `ck_tile::make_alibi_from_lr_mask`
* `ck_tile::make_generic_attention_mask_coordinates_from_lr_window`
* `ck_tile::make_tuple`
* `ck_tile::numeric<T>::max`
* `ck_tile::reference_batched_dropout`
* `ck_tile::reference_batched_dropout_randval`
* `ck_tile::reference_batched_elementwise`
* `ck_tile::reference_batched_gemm`
* `ck_tile::reference_batched_masking`
* `ck_tile::reference_batched_rotary_position_embedding`
* `ck_tile::reference_batched_softmax`
* `ck_tile::reference_unary_elementwise`
* `ck_tile::sqrt`
* `ck_tile::type_convert`