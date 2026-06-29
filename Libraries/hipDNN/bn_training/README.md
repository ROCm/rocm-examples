# hipDNN Batch Normalization Training Example

## Description

This example executes the forward pass of a batch normalization training graph on a 4D input tensor.

For an input `x` of shape `(N, C, H, W)`, the mean and variance are calculated over the `N`, `H`, and `W` dimensions for each of the `C` channels, resulting in a `mean` and `inv_variance` of shape `(1, C, 1, 1)`. It then transforms the input and updates the running statistics:

```python
y = scale * ((x - mean) * inv_variance) + bias
next_running_mean = (1 - momentum) * prev_running_mean + momentum * batch_mean
next_running_variance = (1 - momentum) * prev_running_variance + momentum * batch_variance
```

The graph outputs the normalized tensor `y`, along with the batch mean/variance (`mean`, `inv_variance`) required for the backward pass, and the updated population statistics (`next_running_mean`, `next_running_variance`) required for inference.

The application supports two modes selectable via command line flags: batch-stats-only (`--batch-stats-only`), which computes batch statistics without updating running statistics, and full training (`--full-training`), which computes batch statistics and updates running statistics.

## Application flow

1. Parse command line arguments, including optional `--batch-stats-only` and `--full-training` mode selection.
2. Create a hipDNN handle using the RAII helper.
3. Create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for input (`x`), scale, and bias. Create epsilon as a pass-by-value scalar.
5. Configure `BatchnormAttributes` with epsilon. If full-training mode is selected, create previous running statistics tensors and momentum scalar, and call `set_previous_running_stats()`.
6. Add the batchnorm training node to the graph. Mark output tensors (`y`, `savedMean`, `savedInvVariance`, and optionally `nextRunningMean`, `nextRunningVariance`).
7. Build the graph against the hipDNN handle.
8. Allocate host tensors and initialize with random values.
9. Create a variant pack mapping tensor UIDs to device memory pointers.
10. Execute the graph and synchronize results to host memory.
11. Print sample output values and optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. A `graph::Graph` is created, configured with data types, and built against a hipDNN handle before execution.
- `graph::BatchnormAttributes` configures the batch normalization training operation, including epsilon and optional running statistics updates.
- The `set_epsilon()` method accepts a pass-by-value scalar via `TensorAttributes::set_value()`.
- The `set_previous_running_stats()` method enables full training mode by providing previous running statistics and momentum. When omitted, the graph operates in batch-stats-only mode.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_intermediate_data_type()`
- `graph->set_compute_data_type()`
- `graph::BatchnormAttributes`
- `BatchnormAttributes::set_epsilon()`
- `BatchnormAttributes::set_previous_running_stats()`
- `graph::TensorAttributes`
- `TensorAttributes::set_value()`
- `graph->batchnorm()`
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
