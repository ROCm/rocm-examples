# hipDNN Fused Batch Normalization Inference and Activation Example

## Description

This example executes a fused batch normalization inference and activation graph.

The fused graph consists of two operations:

1. **Batchnorm Inference**: Normalizes input `x` using saved statistics (mean and inverse variance).

   ```python
   bn_y = scale * ((x - mean) * inv_variance) + bias
   ```

2. **Activation (ReLU)**: Applies ReLU activation.

   ```python
   y = relu(bn_y) = max(bn_y, 0)
   ```

Uses `CpuReferenceGraphExecutor` for validation by serializing the graph with `graph->to_binary()`.

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for input (`x`), scale, bias, mean, and inverse variance.
5. Add a `batchnorm_inference` node followed by a `pointwise` node with `PointwiseMode::RELU_FWD` to the graph.
6. Mark the activated output tensor and build the graph.
7. Allocate host tensors and initialize with random values.
8. Create a variant pack, query workspace size, and allocate workspace memory.
9. Execute the graph with the variant pack and workspace.
10. Synchronize results to host memory and print sample output values.
11. Optionally validate using `CpuReferenceGraphExecutor` with the serialized graph from `graph->to_binary()`.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. Multiple operations are chained together in a single graph for potential fusion by the backend.
- `graph::BatchnormInferenceAttributes` configures the batch normalization inference node using pre-computed mean and inverse variance.
- `graph::PointwiseAttributes` with `PointwiseMode::RELU_FWD` configures the ReLU activation applied to the batchnorm output.
- The intermediate batchnorm output tensor is implicitly virtual (not marked as output), allowing the backend to optimize memory usage.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_intermediate_data_type()`
- `graph->set_compute_data_type()`
- `graph::BatchnormInferenceAttributes`
- `graph->batchnorm_inference()`
- `graph::PointwiseAttributes`
- `PointwiseAttributes::set_mode(PointwiseMode::RELU_FWD)`
- `graph->pointwise()`
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
- `PointwiseMode::RELU_FWD`
- `TensorLayout`
