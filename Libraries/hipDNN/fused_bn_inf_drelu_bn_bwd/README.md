# hipDNN Fused Batch Normalization Inference, ReLU Backward, and Batch Normalization Backward Example

## Description

This example executes a fused 3-operation graph demonstrating batchnorm inference followed by activation backward and batchnorm backward passes.

The fused graph consists of three operations:

1. **Batchnorm Inference (Forward)**: Normalizes input `x` using saved statistics.

   ```python
   bn_y = scale * ((x - mean) * inv_variance) + bias
   ```

2. **Activation Backward (ReLU)**: Computes gradient through ReLU activation.

   ```python
   dx_drelu[i] = dy[i] if bn_y[i] > 0 else 0
   ```

3. **Batchnorm Backward**: Computes gradients with respect to inputs and parameters.

   ```python
   dbias = sum(dx_drelu)
   x_hat = (x - mean) * inv_variance
   dscale = sum(dx_drelu * x_hat)
   dx = (scale * inv_variance) * (dx_drelu - (dbias + x_hat * dscale) / nhw)
   ```

The intermediate outputs (`bn_y` and `dx_drelu`) are marked as virtual, allowing the backend to optimize memory usage.

The data flow through the fused graph is:

```text
Inputs: x, dy, scale, bias, mean, inv_variance
        |
    bn_y = batchnorm_inference(x, mean, inv_variance, scale, bias)
        | (virtual tensor)
    dx_drelu = activation_backward(bn_y, dy)
        | (virtual tensor)
    [dx, dscale, dbias] = batchnorm_backward(dx_drelu, x, scale, mean, inv_variance)
        |
Outputs: dx, dscale, dbias
```

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for inputs (`x`, `dy`, `scale`, `bias`, `savedMean`, `savedInvVariance`).
5. Add three chained operations to the graph: `batchnorm_inference` (forward), `pointwise` with `PointwiseMode::RELU_BWD` (activation backward), and `batchnorm_backward` (backward). Intermediate tensors are virtual.
6. Mark output tensors (`dx`, `dscale`, `dbias`) and build the graph.
7. Allocate host tensors and initialize with random values.
8. Create a variant pack, query workspace size, and allocate workspace memory.
9. Execute the graph with the variant pack and workspace.
10. Synchronize results to host memory and print sample output values.
11. Optionally validate using `CpuReferenceGraphExecutor` with the serialized graph from `graph->to_binary()`.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. Multiple operations are chained together in a single graph for potential fusion by the backend.
- **Virtual tensors**: Intermediate tensors (`bn_y`, `dx_drelu`) are not marked as output, making them virtual. The backend can optimize memory by not materializing these tensors.
- `graph::BatchnormInferenceAttributes` configures the forward batchnorm inference node.
- `graph::PointwiseAttributes` with `PointwiseMode::RELU_BWD` configures the ReLU backward pass, which gates the upstream gradient by the forward activation output.
- `graph::BatchnormBackwardAttributes` with `set_saved_mean_and_inv_variance()` configures the backward batchnorm node using saved statistics from the forward pass.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_intermediate_data_type()`
- `graph->set_compute_data_type()`
- `graph::BatchnormInferenceAttributes`
- `graph->batchnorm_inference()`
- `graph::PointwiseAttributes`
- `PointwiseAttributes::set_mode(PointwiseMode::RELU_BWD)`
- `graph->pointwise()`
- `graph::BatchnormBackwardAttributes`
- `BatchnormBackwardAttributes::set_saved_mean_and_inv_variance()`
- `graph->batchnorm_backward()`
- `graph->build()`
- `graph->execute()`
- `graph->get_workspace_size()`
- `graph->to_binary()`

### HIP Runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
- `hipDeviceSynchronize`

### Data Types and Enums

- `hipdnnHandle_t`
- `hipdnn_frontend::DataType`
- `PointwiseMode::RELU_BWD`
- `TensorLayout`
