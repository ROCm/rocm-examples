# CK Tile Programming Model: Batched Tensor Contraction

## Description

This example demonstrates how to perform batched tensor contraction operations
using the CK Tile programming model. Batched contraction generalizes matrix multiplication to multi-dimensional tensors, supporting arbitrary contraction
patterns across batch, M, N, and K dimensions.

### Supported architectures

The example is available for
[all supported GPU architectures](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

### Application flow

1. Command line arguments are parsed to configure tensor dimensions across G
   (batch), M, N, and K dimensions.
2. Host memory is allocated for input tensors A and B, auxiliary tensors D, and output 
   tensor E.
3. Input tensors are initialized with random values.
4. Device memory is allocated and input data is copied to the device.
5. CK Tile's batched contraction kernel is instantiated and launched on the
   device.
6. If validation is enabled, the results are compared against a CPU reference 
   implementation.
7. Performance metrics including execution time, TFLOPS, and memory bandwidth are 
   reported.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of the CK Tile programming model's key components:

* A **shape** defines the hierarchical tile structure and memory layout.
* A **problem** combines data types with the shape configuration.
* A **pipeline** schedules the sequence of operations for a kernel.
* A **kernel** implements the actual computation using the problem and pipeline definitions.

### Tensor contraction

The batched contraction operation supports:

* Multi-dimensional batch dimensions (G0, G1, ...)
* Multi-dimensional M dimensions for first operand
* Multi-dimensional N dimensions for second operand
* Multi-dimensional K dimensions for contraction
* Multiple auxiliary D tensors for fusion
* Configurable layouts and strides for all tensors
* Split-K optimization for improved parallelism

## Building

### Linux

Make sure that the dependencies are installed, or use the
[provided Dockerfiles](../../../../Dockerfiles/) to build and run the examples
in a containerized environment that has all prerequisites installed.

```shell
cd Libraries/ComposableKernel/contraction/batched_contraction
cmake -S . -B build
cmake --build build
```

## Running

```shell
./build/ComposableKernel_ck_tile_batched_contraction [options]
```

### Command line arguments

* `-g_dims` - Batch dimensions (comma-separated, default: "2")
* `-m_dims` - M dimensions (comma-separated, default: "512")
* `-n_dims` - N dimensions (comma-separated, default: "512")
* `-k_dims` - K dimensions (comma-separated, default: "512")
* `-a_layout` - Tensor A layout (default: R)
* `-b_layout` - Tensor B layout (default: C)
* `-d_layout` - Tensor D layout (default: R)
* `-e_layout` - Tensor E layout (default: R)
* `-split_k` - Split-K factor for improved parallelism (default: 1)
* `-v` - Validation mode: 0 = No validation, 1 = CPU validation (default: 1)
* `-warmup` - Number of warmup iterations (default: 10)
* `-repeat` - Number of benchmark iterations (default: 100)
