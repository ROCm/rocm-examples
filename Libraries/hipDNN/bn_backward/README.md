# hipDNN Batch Normalization Backward Example

## Description

This example executes the backward pass of a batch normalization graph to compute gradients of the loss function.

Given the upstream gradient `dy` of shape `(N, C, H, W)`, the downstream learnable gradients are computed with the chain-rule over the batch and spatial dimensions (`N, H, W`) using saved batch statistics:

```python
dbias = sum(dy)
x_hat = (x - mean) * inv_variance
dscale = sum(dy * x_hat)
dx = (scale * inv_variance) * (dy - (dbias + x_hat * dscale) / nhw)
```

where `nhw = N * H * W` is the number of elements averaged per channel.

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for input gradient (`dy`), input (`x`), scale, saved mean, and saved inverse variance.
5. Configure `BatchnormBackwardAttributes` with saved mean and inverse variance, and add the backward batchnorm node to the graph.
6. Mark output tensors (`dx`, `dscale`, `dbias`) and build the graph.
7. Allocate host tensors and initialize with random values.
8. Create a variant pack mapping tensor UIDs to device memory pointers.
9. Execute the graph and synchronize results to host memory.
10. Print sample output values and optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. A `graph::Graph` is created, configured with data types, and built against a hipDNN handle before execution.
- `graph::BatchnormBackwardAttributes` configures the backward batch normalization operation. The `set_saved_mean_and_inv_variance()` method provides the saved batch statistics from the forward pass.
- The graph returns a structured binding of three gradients: `dx` (input gradient), `dscale` (scale gradient), and `dbias` (bias gradient).

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_intermediate_data_type()`
- `graph->set_compute_data_type()`
- `graph::BatchnormBackwardAttributes`
- `BatchnormBackwardAttributes::set_saved_mean_and_inv_variance()`
- `graph->batchnorm_backward()`
- `graph->build()`
- `graph->execute()`

### HIP Runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
- `hipDeviceSynchronize`

### Data Types and Enums

- `hipdnnHandle_t`
- `hipdnn_frontend::DataType`
- `TensorLayout`
