# hipDNN Batch Normalization Inference Example

## Description

This example executes a single-node batch normalization inference graph on a 4D input tensor using inverse variance.

It normalizes each dimension of the input tensor `x` of shape `(N, C, H, W)`, using pre-calculated population statistics (mean and inverse variance). The result is then transformed by the learned parameters `scale` and `bias`, each with shape `(1, C, 1, 1)`:

```python
y = scale * ((x - mean) * inv_variance) + bias
```

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for input (`x`), scale, bias, mean, and inverse variance.
5. Configure `BatchnormInferenceAttributes` and add the inference batchnorm node to the graph.
6. Mark the output tensor (`y`) and build the graph.
7. Allocate host tensors and initialize with random values.
8. Create a variant pack mapping tensor UIDs to device memory pointers.
9. Execute the graph and synchronize results to host memory.
10. Print sample output values and optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. A `graph::Graph` is created, configured with data types, and built against a hipDNN handle before execution.
- `graph::BatchnormInferenceAttributes` configures the batch normalization inference operation, including providing the pre-computed mean and inverse variance tensors.
- The `set_io_data_type()`, `set_intermediate_data_type()`, and `set_compute_data_type()` methods control precision for input/output, intermediate, and compute operations respectively.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_intermediate_data_type()`
- `graph->set_compute_data_type()`
- `graph::BatchnormInferenceAttributes`
- `graph->batchnorm_inference()`
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
