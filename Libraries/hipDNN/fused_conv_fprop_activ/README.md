# hipDNN Fused Convolution Forward and Activation Example

## Description

This example executes a fused convolution forward pass with activation function in a single graph.

The fused graph consists of two operations:

1. **Convolution Forward**: Performs standard 2D convolution.

   ```python
   conv_y = conv(x, w, stride, padding, dilation)
   ```

2. **Activation (Clamped ReLU)**: Applies ReLU activation with upper and lower clipping bounds to the convolution output.

   ```python
   y = clamp(relu(conv_y), lower_clip, upper_clip) = min(max(conv_y, lower_clip), upper_clip)
   ```

The intermediate convolution output (`conv_y`) is marked as virtual, allowing the backend to optimize memory usage and potentially fuse operations.

The data flow through the fused graph is:

```text
Inputs: x (input tensor), w (filter weights)
          |
     conv_y = convolution_forward(x, w, stride, padding, dilation)
          | (virtual tensor)
     y = activation_forward(conv_y, mode=RELU, lower_clip=0.2, upper_clip=0.7)
          |
Output: y (activated convolution result)
```

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for input (`x`) and filter (`w`).
5. Add a `conv_fprop` node followed by a `pointwise` node with `PointwiseMode::RELU_FWD` and clamping bounds to the graph. The convolution output is marked as virtual.
6. Mark the final output tensor and build the graph.
7. Allocate host tensors and initialize with random values.
8. Create a variant pack, query workspace size, and allocate workspace memory.
9. Execute the graph with the variant pack and workspace.
10. Synchronize results to host memory and print sample output values.
11. Optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. Multiple operations are chained together in a single graph for potential fusion by the backend.
- `graph::ConvFpropAttributes` configures the convolution forward operation with padding, stride, and dilation.
- `graph::PointwiseAttributes` with `PointwiseMode::RELU_FWD` configures the activation. The `set_relu_lower_clip()` and `set_relu_upper_clip()` methods set the clamping bounds.
- The convolution output is not marked as output (`set_output(false)`), making it a virtual tensor that does not require separate memory allocation.

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
- `PointwiseAttributes::set_mode(PointwiseMode::RELU_FWD)`
- `PointwiseAttributes::set_relu_lower_clip()`
- `PointwiseAttributes::set_relu_upper_clip()`
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
- `PointwiseMode::RELU_FWD`
- `TensorLayout`
