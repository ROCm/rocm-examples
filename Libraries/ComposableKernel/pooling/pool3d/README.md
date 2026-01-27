# CK Tile Programming Model: 3D Pooling

## Description

This example demonstrates how to perform 3D pooling operations using the CK Tile programming model. The pooling kernel supports both 2D and 3D pooling operations for downsampling feature maps in neural networks.

### Supported architectures

The example is available for
[all supported GPU architectures](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

### Application flow

1. Command line arguments are parsed to configure input dimensions, pooling window size, stride, dilation, and padding parameters.
2. Host memory is allocated for input and output tensors.
3. Input tensor is initialized with random values.
4. Device memory is allocated and input data is copied to the device.
5. CK Tile's 3D pooling kernel is instantiated and launched on the device.
6. If validation is enabled, the results are compared against a CPU reference implementation.
7. Performance metrics including execution time are reported.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The example makes use of the CK Tile programming model's key components:

* A **shape** defines the hierarchical tile structure and memory layout.
* A **problem** combines data types with the shape configuration.
* A **pipeline** schedules the sequence of operations for a kernel.
* A **kernel** implements the actual computation using the problem and pipeline definitions.

### Pooling operations

The pooling kernel supports:

* Configurable pooling window sizes in depth, height, and width dimensions
* Adjustable stride values for each dimension
* Dilation support for expanded receptive fields
* Padding configuration for boundary handling

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfiles](../../../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

```shell
cd Libraries/ComposableKernel/pooling/pool3d
cmake -S . -B build
cmake --build build
```

## Running

```shell
./build/ComposableKernel_ck_tile_pool3d [options]
```

### Command line arguments

* `-N` - Batch size (default: 2)
* `-D` - Depth dimension (default: 30)
* `-H` - Height dimension (default: 30)
* `-W` - Width dimension (default: 30)
* `-C` - Channel dimension (default: 32)
* `-Z` - Pooling window depth (default: 2)
* `-Y` - Pooling window height (default: 2)
* `-X` - Pooling window width (default: 2)
* `-Sz` - Window stride depth (default: 2)
* `-Sy` - Window stride height (default: 2)
* `-Sx` - Window stride width (default: 2)
* `-Dz` - Window dilation depth (default: 1)
* `-Dy` - Window dilation height (default: 1)
* `-Dx` - Window dilation width (default: 1)
* `-LeftPz` - Left padding depth (default: 1)
* `-LeftPy` - Left padding height (default: 1)
* `-LeftPx` - Left padding width (default: 1)
* `-RightPz` - Right padding depth (default: 1)
* `-RightPy` - Right padding height (default: 1)
* `-RightPx` - Right padding width (default: 1)
* `-v` - Validation mode: 0 = No validation, 1 = CPU validation (default: 1)
* `-warmup` - Number of warmup iterations (default: 0)
* `-repeat` - Number of benchmark iterations (default: 1)
