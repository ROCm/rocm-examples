# CK Tile Framework: Getting Started with Tile Copy Operations

## Description

This is a minimal CK Tile memory copy implementation demonstrating the basic
setup required to write a kernel in CK Tile. This experimental kernel is
intended for novice CK developers. It introduces the building blocks of CK Tile
and provides a sandbox for experimenting with kernel parameters.

### Supported architectures

This example works with
[all supported GPU architectures](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

### Application flow

1. Command line arguments are parsed to configure matrix dimensions and
   execution parameters.
2. Host memory is allocated for input and output tensors.
3. The input tensor is initialized with random values.
4. Device memory is allocated and input data is copied to the device.
5. CK Tile's copy kernel is instantiated and launched on the device.
6. If validation is enabled, the results are compared against the input data.
7. Performance metrics, including execution time, are reported.
8. All buffers are freed automatically.

## Key APIs and concepts

### CK Tile architecture

The CK Tile framework is built around four key architectural components:

* The **shape** defines the hierarchical tile structure and memory layout.
* The **problem** combines data types with the shape configuration.
* The **pipeline** schedules the sequence of operations for a kernel.
* The **kernel** implements the actual computation using the problem and pipeline definitions.

### Hierarchical tile structure

The CK Tile framework organizes work hierarchically:

1. **ThreadTile** - Number of contiguous elements processed by a single thread
  (enables vectorized loads/stores)
2. **WaveTile** - Number of elements covered by a single wave (64 threads on
   CDNA, 32 threads on RDNA)
3. **BlockTile** - Number of elements covered by one block (typically mapped to one CU)
4. **BlockWaves** - Number of concurrent waves active in a block

For more information on CK Tile terminology, refer to the
[Composable Kernel Glossary](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/reference/Composable-Kernel-Glossary.html).

### Copy kernel implementation

Copy kernel demonstrates:

* Creating tensor views with specified dimensions and strides
* Instantiating tile windows that provide views into specific tensor regions
* Loading global memory data into registers
* Storing register data in global memory
* Moving tile windows to process larger tensors

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfiles](../../../../Dockerfiles/) to build and run the examples in a containerized
environment that has all prerequisites installed.

```shell
cd Libraries/ComposableKernel/basic/copy
cmake -S . -B build
cmake --build build
```

## Running

```shell
./build/ComposableKernel_ck_tile_copy [options]
```

### Command line arguments

* `-m` - Input matrix rows (default: 64)
* `-n` - Input matrix columns (default: 8)
* `-id` - Wave to use for computation (default: 0)
* `-v` - Validation flag to check device results (default: 1)
* `-prec` - Datatype precision to use (default: fp16)
* `-warmup` - Number of warmup iterations (default: 50)
* `-repeat` - Number of iterations for kernel execution time (default: 100)
