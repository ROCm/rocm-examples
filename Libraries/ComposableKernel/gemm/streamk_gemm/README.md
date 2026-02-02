# CK Tile Programming Model: Stream-K GEMM

## Description

This example demonstrates how to perform a Stream-K GEMM operation using CK
Tile. Stream-K is an advanced GEMM algorithm that provides improved load
balancing and performance for matrix multiplication operations.

### Supported architectures

This example works with
[all supported GPU architectures](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and execution parameters.
2. Buffers for the input matrices (A and B) and the output matrix (C) are
   created on the host.
3. The input matrices are initialized with random values based on the specified initialization strategy.
4. Buffers are created on the device and input data is copied to the device.
5. CK Tile's Stream-K GEMM kernel is instantiated and launched on the device.
6. If validation is enabled, the results are compared against a CPU or GPU reference implementation.
7. Performance metrics including execution time, TFLOPS, and memory bandwidth are reported.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of CK Tile's key components:

* The **shape** defines the hierarchical tile structure and memory layout.
* The **problem** combines data types with the shape configuration.
* The **pipeline** schedules the sequence of operations for a kernel, including Stream-K specific optimizations.
* The **kernel** implements the actual computation using the problem and pipeline definitions.

### Stream-K optimization

Stream-K improves upon traditional tiled GEMM by:

* Better load balancing across GPU compute units
* Reduced idle time through work redistribution
* Support for persistent and non-persistent execution modes

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfiles](../../../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

```shell
cd Libraries/ComposableKernel/gemm/streamk_gemm
cmake -S . -B build
cmake --build build
```

## Running

```shell
./build/ComposableKernel_ck_tile_streamk_gemm [options]
```

### Command line arguments

* `-m` - M dimension (default: 512)
* `-n` - N dimension (default: 512)
* `-k` - K dimension (default: 512)
* `-a_layout` - Tensor A data layout (default: R)
* `-b_layout` - Tensor B data layout (default: C)
* `-c_layout` - Tensor C data layout (default: R)
* `-reduction_strategy` - Strategy for storing results in C tensor: atomic/reduction (default: atomic)
* `-persistent_dp` - Persistent strategy for data-parallel section: 0 = non-persistent, 1 = persistent (default: 0)
* `-stride_a` - Tensor A stride (default: 0)
* `-stride_b` - Tensor B stride (default: 0)
* `-stride_c` - Tensor C stride (default: 0)
* `-v` - Validation strategy: 0 = No validation, 1 = CPU validation, 2 = GPU validation (default: 1)
* `-prec` - Data type: fp16/bf16/fp8/bf8 (default: fp16)
* `-warmup` - Number of warmup iterations (default: 50)
* `-repeat` - Number of benchmark iterations (default: 100)
* `-timer` - Timing mode: gpu = GPU timer, cpu = CPU timer (default: gpu)
* `-init` - Data initialization strategy: 0 = random, 1 = linear, 2 = constant(1) (default: 0)
* `-flush_cache` - Flush the cache before running the kernel (default: true)
