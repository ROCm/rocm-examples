# CK Tile Programming Model: Fused Multi-Head Attention Forward v3 example

## Description

This example demonstrates how to perform FMHA (fused multi-head attention)
forward pass with CK Tile using the v3 implementation. This version focuses
exclusively on the forward pass and includes support for variable-length
sequences with padding, allowing efficient processing of batches where sequences
have different effective lengths.

### Supported architectures

The example works on the `gfx950` architecture only.

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and
   execution parameters.
2. Buffers for the Q, K, V tensors are created on the host and initialized with
   random values.
3. Buffers for all tensors are created on the device.
4. The tensors are copied from the host to the device.
5. Optional variable-length sequence support is configured: If effective
   sequence lengths are provided via `q_eff_lens` or `kv_eff_lens`, cumulative
   sequence length arrays are computed and uploaded to the device. These arrays
   allow the kernel to skip padded regions and only process valid data.
6. CK Tile's `fmha_fwd_v3` kernel is instantiated and launched on the device.
7. If validation is enabled, the results are compared against a custom reference
   implementation: For variable-length sequences, the reference computation
   processes only the effective (non-padded) portion of each sequence. Padded
   regions are zero-initialized and verified to remain zero.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of three key architectural components:

* The **shape** defines the hierarchical tile structure and memory layout. In this example it is set to a custom
  shape based on the sequence length.
* The **problem** combines data types with the shape configuration. In this example its components are set individually
  by specifying the individual tensors' dimensions, strides, etc. as part of the arguments to the kernel.
* The **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `fmha_fwd_v3` is used.

### Variable-length sequence support

The v3 implementation supports efficient handling of variable-length sequences through cumulative sequence length arrays:

* `cu_seqlen_q_ptr`: Cumulative sequence lengths for queries (size: batch + 1)
* `cu_seqlen_kv_ptr`: Cumulative sequence lengths for keys/values (size: batch + 1)

When provided, these arrays allow the kernel to:

* Process only the valid (non-padded) portion of each sequence
* Skip computation on padded regions
* Maintain correct attention masks per sequence

This is particularly useful for:

* Batches with sequences of varying lengths
* Reducing unnecessary computation on padding
* Maintaining accuracy when sequences have different effective lengths

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::bf16_t`
* `ck_tile::DeviceMem`
* `ck_tile::FillNormalDistribution`
* `ck_tile::fmha_fwd_v3_args`
* `ck_tile::fp16_t`
* `ck_tile::HostTensor`
* `ck_tile::identity`
* `ck_tile::index_t`
* `ck_tile::scales`
* `ck_tile::stream_config`

#### Functions

* `ck_tile::check_err`
* `ck_tile::fmha_fwd_v3`
* `ck_tile::make_generic_attention_mask_from_lr_window`
* `ck_tile::make_tuple`
* `ck_tile::reference_batched_gemm`
* `ck_tile::reference_batched_masking`
* `ck_tile::reference_batched_softmax`
* `ck_tile::sqrt`
