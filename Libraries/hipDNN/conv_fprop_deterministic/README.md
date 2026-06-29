# hipDNN Deterministic Convolution Forward Example

## Description

This example executes a deterministic forward pass of a 2D convolution operation. It specifically targets the deterministic engine variant (`MIOPEN_ENGINE_DETERMINISTIC`), which guarantees bit-reproducible results across runs at a potential performance cost. This is useful for debugging and validation scenarios where exact reproducibility is required.

The example executes the same convolution twice with identical inputs and verifies that the outputs are bit-exact.

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Initialize frontend logging with `initializeFrontendLogging()` and create a hipDNN handle with `hipdnnCreate()`.
3. Create a `graph::Graph` and configure IO and compute data types.
4. Set the preferred engine to deterministic mode with `graph->set_preferred_engine_id_ext()`.
5. Create tensor attributes for input (`x`) and filter (`w`) tensors.
6. Configure `ConvFpropAttributes` with padding, stride, and dilation, and add the forward convolution node to the graph.
7. Mark the output tensor (`y`) and build the graph.
8. Allocate host tensors and initialize with random values.
9. Query workspace size and allocate workspace memory.
10. Execute the graph twice with identical inputs into separate output buffers.
11. Copy results to host and verify that both runs produce bit-exact results.
12. Optionally validate against CPU reference.
13. Clean up with `hipdnnDestroy()`.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. A `graph::Graph` is created, configured with data types and a preferred engine, then built against a hipDNN handle before execution.
- `graph->set_preferred_engine_id_ext()` selects the deterministic engine variant, which trades performance for bit-exact reproducibility across multiple executions.
- `graph::ConvFpropAttributes` configures the convolution forward operation, identical to the standard convolution forward example.
- The example uses explicit hipDNN handle management with `hipdnnCreate()` and `hipdnnDestroy()` instead of the RAII helper, along with `initializeFrontendLogging()` for diagnostic output.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_compute_data_type()`
- `graph->set_preferred_engine_id_ext()`
- `graph::ConvFpropAttributes`
- `ConvFpropAttributes::set_padding()`
- `ConvFpropAttributes::set_stride()`
- `ConvFpropAttributes::set_dilation()`
- `graph->conv_fprop()`
- `graph->build()`
- `graph->execute()`
- `graph->get_workspace_size()`
- `hipdnnCreate`
- `hipdnnDestroy`
- `initializeFrontendLogging`

### HIP Runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
- `hipDeviceSynchronize`

### Data Types and Enums

- `hipdnnHandle_t`
- `hipdnn_frontend::DataType`
- `TensorLayout`
- `MIOPEN_ENGINE_DETERMINISTIC`
