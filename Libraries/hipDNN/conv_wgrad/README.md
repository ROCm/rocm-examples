# hipDNN Convolution Backward Filter Example

## Description

This example executes the backward pass (filter gradient) of a 2D convolution operation to compute filter gradients.

For an output gradient tensor `dy` of shape `(N, K, H_out, W_out)` and an input tensor `x` of shape `(N, C, H_in, W_in)`, the convolution backward filter operation computes the filter gradient tensor `dw` of shape `(K, C, R, S)`:

```python
dw[k, c, r, s] = sum(sum(sum(dy[n, k, p, q] * x[n, c, h, w]
                             for n in range(N))
                         for p in range(H_out))
                     for q in range(W_out))
```

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO and compute data types.
4. Compute output gradient dimensions from input dimensions and convolution parameters.
5. Create tensor attributes for the output gradient (`dy`) and input (`x`) tensors.
6. Configure `ConvWgradAttributes` with pre-padding, post-padding, stride, and dilation, and add the backward filter convolution node to the graph.
7. Mark the output tensor (`dw`) and build the graph.
8. Allocate host tensors and initialize with random values.
9. Create a variant pack, query workspace size, and allocate workspace memory.
10. Execute the graph with the variant pack and workspace.
11. Synchronize results to host memory and print sample output values.
12. Optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. A `graph::Graph` is created, configured with data types, and built against a hipDNN handle before execution.
- `graph::ConvWgradAttributes` configures the convolution backward filter operation, including pre-padding, post-padding, stride, and dilation parameters.
- Like `ConvDgradAttributes`, this uses `set_pre_padding()` and `set_post_padding()` to specify asymmetric padding.
- Workspace memory is required for convolution operations. The required size is queried with `graph->get_workspace_size()` and allocated before calling `graph->execute()`.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_compute_data_type()`
- `graph::ConvWgradAttributes`
- `ConvWgradAttributes::set_pre_padding()`
- `ConvWgradAttributes::set_post_padding()`
- `ConvWgradAttributes::set_stride()`
- `ConvWgradAttributes::set_dilation()`
- `graph->conv_wgrad()`
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
