# CK Tile Programming Model: Multiple ABD GEMM

## Description

This example demonstrates how to perform a Multiple ABD GEMM operation using CK
Tile. The Multiple ABD GEMM performs matrix multiplication with multiple
auxiliary input tensors, allowing for more complex fusion patterns.

### Supported architectures

The example works with
[all supported GPU architectures](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for the input matrices (A, B), auxiliary tensor (D), and output
   matrix (E) are created on the host.
3. The input matrices and auxiliary tensors are initialized with random values.
4. Buffers for the input matrices (A, B), auxiliary tensor (D), and output
   matrix (E) are created on the device. A, B, and D are copied to the
   device while E is initialized to 0.
5. CK Tile's Multiple ABD GEMM kernel is instantiated and launched on the device.
6. If validation is enabled, the results are compared against a CPU reference implementation.
7. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use CK Tile's key components:

* The **shape** defines the hierarchical tile structure and memory layout.
* The **problem** combines data types with the shape configuration.
* The **pipeline** schedules the sequence of operations for a kernel.
* The **kernel** implements the actual computation using the problem and pipeline definitions.

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfiles](../../../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

```shell
cd Libraries/ComposableKernel/gemm/gemm_multi_abd
cmake -S . -B build
cmake --build build
```

## Running

```shell
./build/ComposableKernel_ck_tile_gemm_multi_abd [options]
```

### Command line arguments

* `-m` - M dimension (default: 3840)
* `-n` - N dimension (default: 4096)
* `-k` - K dimension (default: 4096)
* `-as_layout` - Tensor A layout (default: R)
* `-bs_layout` - Tensor B layout (default: C)
* `-ds_layout` - Tensor D layout (default: R)
* `-e_layout` - Tensor E layout (default: R)
* `-stride_as` - Tensor A strides (default: 0)
* `-stride_bs` - Tensor B strides (default: 0)
* `-stride_e` - Tensor E strides (default: 0)
* `-stride_ds` - Tensor D strides (default: 0)
* `-validate` - Validation mode: 0 = No validation, 1 = GPU validation (default: 1)
* `-warmup` - Number of warmup iterations (default: 10)
* `-repeat` - Number of benchmark iterations (default: 100)
* `-kbatch` - kbatch for SplitK (default: 1)
