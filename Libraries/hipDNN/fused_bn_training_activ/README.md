# hipDNN Fused Batch Normalization Training and Activation Example

## Description

This example executes a fused batch normalization training and activation graph.

The fused graph consists of two operations:

1. **Batchnorm Training**: Normalizes input `x` using batch statistics, updates running statistics (optional), and outputs saved mean and inverse variance.

   ```python
   y_bn = scale * ((x - mean) * inv_variance) + bias
   ```

2. **Activation (ReLU)**: Applies ReLU activation.

   ```python
   y = relu(y_bn) = max(y_bn, 0)
   ```

The application supports two modes selectable via command line flags: batch-stats-only (`--batch-stats-only`), which computes batch statistics without updating running statistics, and full training (`--full-training`), which computes batch statistics and updates running statistics. Uses `CpuReferenceGraphExecutor` for validation by serializing the graph with `graph->to_binary()`.

## Application flow

1. Parse command line arguments, including optional `--batch-stats-only` and `--full-training` mode selection.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for input (`x`), scale, and bias. Create epsilon as a pass-by-value scalar.
5. Configure `BatchnormAttributes` with epsilon. If full-training mode, create previous running statistics and momentum tensors and call `set_previous_running_stats()`.
6. Add a `batchnorm` training node followed by a `pointwise` node with `PointwiseMode::RELU_FWD` to the graph.
7. Mark output tensors (`activatedY`, `savedMean`, `savedInvVariance`, and optionally running statistics) and build the graph.
8. Allocate host tensors and initialize with random values.
9. Create a variant pack, query workspace size, and allocate workspace memory.
10. Execute the graph with the variant pack and workspace.
11. Synchronize results to host memory and print sample output values.
12. Optionally validate using `CpuReferenceGraphExecutor` with the serialized graph from `graph->to_binary()`.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. Multiple operations are chained together in a single graph for potential fusion by the backend.
- `graph::BatchnormAttributes` configures the batch normalization training node, with optional running statistics via `set_previous_running_stats()`.
- `graph::PointwiseAttributes` with `PointwiseMode::RELU_FWD` configures the ReLU activation applied to the batchnorm output.
- `graph->to_binary()` serializes the graph for use with `CpuReferenceGraphExecutor` during validation.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_intermediate_data_type()`
- `graph->set_compute_data_type()`
- `graph::BatchnormAttributes`
- `BatchnormAttributes::set_epsilon()`
- `BatchnormAttributes::set_previous_running_stats()`
- `graph->batchnorm()`
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
