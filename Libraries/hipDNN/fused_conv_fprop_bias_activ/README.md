# hipDNN Fused Convolution Forward, Bias, and Activation Example

## Description

This example executes a fused convolution forward pass with bias addition and activation function in a single graph.

The fused graph consists of three operations:

1. **Convolution Forward**: Performs standard 2D convolution.
2. **Pointwise Add (Bias)**: Adds a per-channel bias vector to the convolution output.
3. **Activation (ReLU)**: Applies ReLU activation to the result.

This demonstrates a common deep learning pattern where convolution, bias, and activation are fused into a single graph for improved performance.

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for input (`x`), filter (`w`), and per-channel bias.
5. Add three chained operations to the graph: `conv_fprop`, `pointwise` with `PointwiseMode::ADD` (bias addition), and `pointwise` with `PointwiseMode::RELU_FWD` (activation). Intermediate tensors are virtual.
6. Mark the final output tensor and build the graph.
7. Allocate host tensors and initialize with random values.
8. Create a variant pack, query workspace size, and allocate workspace memory.
9. Execute the graph with the variant pack and workspace.
10. Synchronize results to host memory and print sample output values.
11. Optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. Multiple operations are chained together in a single graph for potential fusion by the backend.
- `graph::ConvFpropAttributes` configures the convolution forward operation with padding, stride, and dilation.
- `graph::PointwiseAttributes` with `PointwiseMode::ADD` configures the bias addition. The bias tensor shape `(1, K, 1, 1)` is derived from the convolution output shape using `getDerivedShape()`, enabling per-channel broadcasting.
- `graph::PointwiseAttributes` with `PointwiseMode::RELU_FWD` configures the ReLU activation applied after bias addition.
- The `set_compute_data_type()` method on both the graph and the bias pointwise attributes ensures consistent compute precision across operations.
- Intermediate tensors (convolution output and bias output) are virtual, requiring no separate memory allocation.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_intermediate_data_type()`
- `graph->set_compute_data_type()`
- `graph::ConvFpropAttributes`
- `ConvFpropAttributes::set_padding()`
- `ConvFpropAttributes::set_stride()`
- `ConvFpropAttributes::set_dilation()`
- `graph->conv_fprop()`
- `graph::PointwiseAttributes`
- `PointwiseAttributes::set_mode(PointwiseMode::ADD)`
- `PointwiseAttributes::set_mode(PointwiseMode::RELU_FWD)`
- `PointwiseAttributes::set_compute_data_type()`
- `graph->pointwise()`
- `graph->build()`
- `graph->execute()`
- `graph->get_workspace_size()`

### HIP Runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
- `hipDeviceSynchronize`

### Data Types and Enums

- `hipdnnHandle_t`
- `hipdnn_frontend::DataType`
- `PointwiseMode::ADD`
- `PointwiseMode::RELU_FWD`
- `TensorLayout`
