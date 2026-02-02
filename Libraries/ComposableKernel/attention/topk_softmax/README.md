# CK Tile Programming Model: topk-softmax example

## Description

This example demonstrates how to launch a topk-softmax kernel with the CK Tile
programming model. The input is a `token*expert` 2D matrix. The operation will
apply one activation function (either softmax or sigmoid) per row (`expert`),
then find the `topk` value for each row. The output is a `token*topk` weight
tensor (usually single-precision) and an index tensor (32-bit integer).
Supported input data types include `fp16` and `bf16`.

### Supported architectures

This example works with
[all supported GPU architectures](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus).

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for the input matrix, the weight and the index tensor are created on the host.
3. The input matrix is initialized with random floating-point values.
4. Buffers for input matrix, the weight tensor and the index tensor are created on the device.
5. The input matrix is copied to the device.
6. CK Tile's `TopkSoftmaxKernel` is instantiated and launched on the device.
7. The weight and index tensors are copied to the host.
8. If validation is enabled, the results are compared against an implementation using CK Tile's `reference_softmax` and
   `reference_topk` functions.
9. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The CK Tile framework is built around four key architectural components:

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
* `ck_tile::DeviceMem`
* `ck_tile::FillUniformDistribution_Unique`
* `ck_tile::fp16_t`
* `ck_tile::HostTensor`
* `ck_tile::index_t`
* `ck_tile::stream_config`
* `ck_tile::TopkSoftmaxHostArgs`
* `ck_tile::TopkSoftmaxKernel`
* `ck_tile::TopkSoftmaxWarpPerRowPipeline`
* `ck_tile::TopkSoftmaxWarpPerRowProblem`

#### Functions

* `ck_tile::check_err`
* `ck_tile::launch_kernel`
* `ck_tile::make_kernel`
* `ck_tile::make_tuple`
* `ck_tile::reference_softmax`
* `ck_tile::reference_topk`
