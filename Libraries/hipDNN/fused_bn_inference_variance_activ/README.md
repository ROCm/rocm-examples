# hipDNN Fused Batch Normalization Inference (Variance) and Activation Example

## Description

This example executes a fused batch normalization inference (with variance) and activation graph.

The fused graph consists of two operations:

1. **Batchnorm Inference (with Variance)**: Normalizes input `x` using saved statistics (mean and variance).

   ```python
   bn_y = scale * ((x - mean) / sqrt(variance + epsilon)) + bias
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
4. Create tensor attributes for input (`x`), scale, bias, mean, and variance. Create epsilon as a pass-by-value scalar.
5. Add a `batchnorm_inference_variance_ext` node followed by a `pointwise` node with `PointwiseMode::RELU_FWD` to the graph.
6. Mark the activated output tensor and build the graph.
7. Allocate host tensors and initialize with random values.
8. Create a variant pack, query workspace size, and allocate workspace memory.
9. Execute the graph with the variant pack and workspace.
10. Synchronize results to host memory and print sample output values.
11. Optionally validate using `CpuReferenceGraphExecutor` with the serialized graph from `graph->to_binary()`.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. Multiple operations are chained together in a single graph for potential fusion by the backend.
- `graph::BatchnormInferenceAttributesVarianceExt` configures the batch normalization inference node with variance and epsilon support, as opposed to `BatchnormInferenceAttributes` which expects pre-computed inverse variance.
- The epsilon value is passed as a scalar using `TensorAttributes::set_value()`, rather than as a device buffer.
- `graph::PointwiseAttributes` with `PointwiseMode::RELU_FWD` configures the ReLU activation applied to the batchnorm output.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_intermediate_data_type()`
- `graph->set_compute_data_type()`
- `graph::BatchnormInferenceAttributesVarianceExt`
- `graph::TensorAttributes`
- `TensorAttributes::set_value()`
- `graph->batchnorm_inference_variance_ext()`
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
