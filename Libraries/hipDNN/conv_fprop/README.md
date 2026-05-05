# hipDNN Convolution Forward Example

## Description

This example executes the forward pass of a 2D convolution operation on a 4D input tensor.

For an input tensor `x` of shape `(N, C, H_in, W_in)` and a filter tensor `w` of shape `(K, C, R, S)`, the convolution operation computes the output tensor `y` of shape `(N, K, H_out, W_out)`:

```python
y[n, k, p, q] = sum(sum(sum(x[n, c, h, w] * w[k, c, r, s]
                            for c in range(C))
                        for r in range(R))
                    for s in range(S))
```

where the input spatial indices `(h, w)` are determined by the output position `(p, q)`, stride, padding, and dilation:

```python
h = p * stride_h - pad_h + r * dilation_h
w = q * stride_w - pad_w + s * dilation_w
```

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO and compute data types.
4. Create tensor attributes for input (`x`) and filter (`w`) tensors.
5. Configure `ConvFpropAttributes` with padding, stride, and dilation parameters, and add the forward convolution node to the graph.
6. Mark the output tensor (`y`) and build the graph.
7. Allocate host tensors and initialize with random values.
8. Create a variant pack mapping tensor UIDs to device memory pointers.
9. Query workspace size with `graph->get_workspace_size()` and allocate workspace memory.
10. Execute the graph with the variant pack and workspace.
11. Synchronize results to host memory and print sample output values.
12. Optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. A `graph::Graph` is created, configured with data types, and built against a hipDNN handle before execution.
- `graph::ConvFpropAttributes` configures the convolution forward operation, including padding, stride, and dilation parameters.
- Workspace memory is required for convolution operations. The required size is queried with `graph->get_workspace_size()` and allocated before calling `graph->execute()`.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_compute_data_type()`
- `graph::ConvFpropAttributes`
- `ConvFpropAttributes::set_padding()`
- `ConvFpropAttributes::set_stride()`
- `ConvFpropAttributes::set_dilation()`
- `graph->conv_fprop()`
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
