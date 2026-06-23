# hipDNN Convolution Backward Data Example

## Description

This example executes the backward pass (data gradient) of a 2D convolution operation to compute input gradients.

For an output gradient tensor `dy` of shape `(N, K, H_out, W_out)` and a filter tensor `w` of shape `(K, C, R, S)`, the convolution backward data operation computes the input gradient tensor `dx` of shape `(N, C, H_in, W_in)`:

```python
dx[n, c, h, w] = sum(sum(sum(dy[n, k, p, q] * w[k, c, r, s]
                             for k in range(K))
                         for r in range(R))
                     for s in range(S))
```

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO and compute data types.
4. Compute output gradient dimensions from input dimensions and convolution parameters.
5. Create tensor attributes for the output gradient (`dy`) and filter (`w`) tensors.
6. Configure `ConvDgradAttributes` with pre-padding, post-padding, stride, and dilation, and add the backward data convolution node to the graph.
7. Set the output tensor (`dx`) dimensions to the original forward input shape, mark it as an output, and build the graph.
8. Allocate host tensors and initialize with random values.
9. Create a variant pack, query workspace size, and allocate workspace memory.
10. Execute the graph with the variant pack and workspace.
11. Synchronize results to host memory and print sample output values.
12. Optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. A `graph::Graph` is created, configured with data types, and built against a hipDNN handle before execution.
- `graph::ConvDgradAttributes` configures the convolution backward data operation, including pre-padding, post-padding, stride, and dilation parameters.
- Unlike `ConvFpropAttributes` which uses a single `set_padding()`, the backward data operation uses `set_pre_padding()` and `set_post_padding()` to specify asymmetric padding.
- The `dx` dimensions are set explicitly because convolution backward data output shape is not inferred from `dy`, `w`, and convolution parameters.
- Workspace memory is required for convolution operations. The required size is queried with `graph->get_workspace_size()` and allocated before calling `graph->execute()`.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_compute_data_type()`
- `graph::ConvDgradAttributes`
- `ConvDgradAttributes::set_pre_padding()`
- `ConvDgradAttributes::set_post_padding()`
- `ConvDgradAttributes::set_stride()`
- `ConvDgradAttributes::set_dilation()`
- `graph->conv_dgrad()`
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
- `TensorLayout`
