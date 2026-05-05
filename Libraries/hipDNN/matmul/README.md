# hipDNN Matrix Multiplication (GEMM) Example

## Description

This example executes a batched matrix multiplication (GEMM) using the hipDNN Frontend graph API.

For input tensors `A` of shape `(B, M, K)` and `B` of shape `(B, K, N)`, the matmul computes the output tensor `C` of shape `(B, M, N)`:

```python
C[b, i, j] = sum(A[b, i, k] * B[b, k, j] for k in range(K))
```

The example uses dimensions `batch=2, M=3, K=4, N=5` and runs with `float`, `half`, and `bfloat16` data types.

## Application flow

1. Parse command line arguments and optionally enable CPU reference validation.
2. Create a hipDNN handle using the RAII helper.
3. For each data type (`float`, `half`, `bfloat16`): create a `graph::Graph` and configure IO, intermediate, and compute data types.
4. Create tensor attributes for `A` and `B` using `graph::makeTensorAttributes()`.
5. Configure `MatmulAttributes` and add the matmul node to the graph.
6. Mark the output tensor (`C`) and build the graph.
7. Initialize input tensors with random values.
8. Create a variant pack, query workspace size, and allocate workspace memory.
9. Execute the graph with the variant pack and workspace.
10. Synchronize results to host memory and print sample output values.
11. Optionally validate against CPU reference.

## Key APIs and Concepts

- The [hipDNN Frontend graph API](https://github.com/ROCm/hipDNN) is used to construct and execute the operation graph. A `graph::Graph` is created, configured with data types, and built against a hipDNN handle before execution.
- `graph::MatmulAttributes` configures the matrix multiplication operation.
- Workspace memory is required for matmul operations. The required size is queried with `graph->get_workspace_size()` and allocated before calling `graph->execute()`.

## Demonstrated API Calls

### hipDNN Frontend

- `graph::Graph`
- `graph->set_io_data_type()`
- `graph->set_intermediate_data_type()`
- `graph->set_compute_data_type()`
- `graph::MatmulAttributes`
- `graph::makeTensorAttributes()`
- `graph->matmul()`
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
- `half`
- `bfloat16`
