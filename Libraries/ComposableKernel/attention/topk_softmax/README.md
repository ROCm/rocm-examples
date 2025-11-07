# CK Tile Programming Model: topk-softmax example

## Description

This example demonstrates how to launch a topk-softmax kernel with the CK Tile programming model. The input is a
`token*expert` 2D matrix. The operation will do one softmax per row (`expert`), then find the `topk` value for each row.
The output is a `token*topk` weight tensor (usually single-precision) and an index tensor (32-bit integer).

### Supported architectures

The example is available for
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

The example makes use of three key architectural components:

* A **problem** combines data types with the shape configuration. In this example the shapes are defined by the tokens,
  experts and strides of the tensors. These are passed to CK Tile's `TopkSoftmaxWarpPerRowProblem`.
* A **pipeline** schedules the sequence of operations for a kernel, such as the data loading, computation, and
  storage phases. In this example it is set to CK Tile's `TopkSoftmaxWarpPerRowPipeline`.
* A **kernel** implements the actual computation using the problem and policy definitions. In this example CK Tile's
  `TopkSoftmaxKernel` is used.

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
