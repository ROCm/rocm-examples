# hipDNN Batch Normalization Inference With Variance Example

## Description

This example executes a single-node batch normalization inference with variance graph on a 4D input tensor.

It normalizes each dimension of the input tensor `x` of shape `(N, C, H, W)`, using pre-calculated population statistics. The result is then transformed by the learned parameters `scale` and `bias`, each with shape `(1, C, 1, 1)` to enable broadcasting over the batch (N) and spatial (H, W) dimensions:

```python
y = scale * ((x - mean) / sqrt(variance + epsilon)) + bias
```

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for input (`x`), scale, bias, mean, and variance. Create epsilon as a pass-by-value scalar using `TensorAttributes::set_value()`.
5. Configure `BatchnormInferenceAttributesVarianceExt` and add the inference batchnorm node with variance and epsilon to the graph.
6. Mark the output tensor (`y`) and build the graph.
7. Allocate host tensors and initialize with random values.
8. Create a variant pack mapping tensor UIDs to device memory pointers.
9. Execute the graph and synchronize results to host memory.
10. Print sample output values and optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. A `graph::Graph` is created, configured with data types, and built against a hipDNN handle before execution.
- Unlike `batchnorm_inference()`, this variant accepts **variance** (not inverse variance) and an **epsilon** parameter. The epsilon value is passed as a scalar using `set_value()` on a `TensorAttributes` object, rather than as a device buffer.
- `graph::BatchnormInferenceAttributesVarianceExt` configures the batch normalization inference operation with variance and epsilon support.

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
