# CK Tile Framework: Permute example

## Description

This example shows how to perform a **tensor permutation** with CK Tile. It reproduces the behavior of  

* `torch.permute` – arbitrary axis re-ordering, and  
* `torch.contiguous` – output laid out contiguously in memory,  

with a single GPU kernel that supports **rank ≤ 8** tensors. Peak performance is *not* the primary goal; readability and
generality of the kernel are.

Additionally, there is an an *optimized* permutation for certain rank-7 layouts that are friendly to AMD matrix-core
(MFMA) instructions. That kernel is restricted to `fp16` input/output and specific index patterns (see
"Alternative implementation" below).

The example performs the computation
$\mathbf{Y}_{i_0, i_1, \ldots, i_{n - 1}} = \mathbf{X}_{i_{\pi(0)}, i_{\pi(1)}, \ldots, i_{\pi(n - 1)}}$, where
$\mathbf{X}$ is a tensor of shape $[d_0, d_1, \ldots, d_{n - 1}]$ and $\pi$ is a permutation.

### Application flow

#### Generic implementation

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. The input tensor $\mathbf{X}$ is created on the host and initialized with random integer values in the range
   $[-15, 15]$.
3. The output tensor $\mathbf{Y}$ is created on the host.
4. A buffer for $\mathbf{X}$ is created on the device and $\mathbf{X}$ is copied from the host to the device.
5. A buffer for $\mathbf{Y}$ is created on the device.
6. CK Tile's generic permutation kernel is instantiated for the given problem size and then launched on the device.
7. If validation is enabled, the results are verified against CK Tile's `reference_permute` function.
8. All buffers are freed automatically.

#### Alternative implementation

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. The input tensor $\mathbf{X}$ is created on the host and initialized with random integer values in the range
   $[-15, 15]$.
3. The output tensor $\mathbf{Y}$ is created on the host.
4. A buffer for $\mathbf{X}$ is created on the device and $\mathbf{X}$ is copied from the host to the device.
5. A buffer for $\mathbf{Y}$ is created on the device.
6. The matrix-core optimized kernel is instantiated for the given problem size and then launched on the device.
7. If validation is enabled, the results are verified against CK Tile's `reference_permute` function.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of three key architectural components:

* **Shape** defines the hierarchical tile structure and memory layout and is passed into the application as a
  user-defined parameter.
* **Problem** is only used for the generic implementation. It combines data types with the shape configuration using
  CK Tile's `GenericPermuteProblem`.
* **Kernel** implements the actual computation using the problem definition. The generic implementation uses
  `GenericPermute`, while the alterative implementation uses a custom kernel.

### Tile programming model

Internally, the kernel is performing a **tilewise permute**: Each thread processes a tile (block) of the input, computes
the permuted indices, and writes to the output.

### Tensor operations

* Tensor views are created using `make_naive_tensor_view_packed` to combine memory buffers with tensor descriptors.
* Tile windows are created using `make_tile_window` to provide distributed access to tensor tiles.
* Data is loaded from global memory to registers using `load_tile`.
* Data is stored from registers to global memory using `store_tile`.
* Tiles are distributed across threads using `make_static_tile_distribution` and `make_static_distributed_tensor`.

### Available kernel implementations

The example shows two different kernel implementations:

* `GenericPermute`: The generic permute implementation is provided as part of CK Tile.
* `matrix_core_swizzle_kernel`: The optimized implementation swizzles the tensor to be more friendly for data loading
  for matrix cores.

## Used API surface

### CK Tile

#### Types

* `ck_tile::ArgParser`
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution`
* `ck_tile::fp8_t`
* `ck_tile::GenericPermute`
* `ck_tile::GenericPermuteHostArgs`
* `ck_tile::GenericPermuteProblem`
* `ck_tile::half_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::sequence`
* `ck_tile::stream_config`
* `ck_tile::tile_distribution_encoding`
* `ck_tile::tuple`
* `ck_tile::WarpGemmMfmaF16F16F32M16N16K16`
* `ck_tile::WarpGemmMfmaF16F16F32M32N32K8`

#### Functions

* `ck_tile::dump_permute_json_results`
* `ck_tile::get_warp_size`
* `ck_tile::launch_kernel`
* `ck_tile::load_tile`
* `ck_tile::make_kernel`
* `ck_tile::make_merge_transform`
* `ck_tile::make_naive_tensor_view_packed`
* `ck_tile::make_static_distributed_tensor`
* `ck_tile::make_static_tile_distribution`
* `ck_tile::make_tile_window`
* `ck_tile::reference_permute`
* `ck_tile::store_tile`
* `ck_tile::transform_tensor_view`
