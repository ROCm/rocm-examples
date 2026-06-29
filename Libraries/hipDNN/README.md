# hipDNN Examples

> **Note:** hipDNN is currently in beta. APIs and behavior may change in future releases.

## Summary

The examples in this subdirectory showcase the functionality of the [hipDNN](https://github.com/ROCm/hipDNN) library. hipDNN provides a graph-based API for constructing and executing deep neural network operations on AMD GPUs, including convolution, batch normalization, and fused operation graphs. The examples build on Linux for the ROCm (AMD GPU) backend.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.25.2)
- Or GNU Make - available via the distribution's package manager.
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 6.x.x)
- [hipDNN](https://github.com/ROCm/hipDNN): `hipdnn_frontend` and `hipdnn_test_sdk` packages.

### Windows

Support for Windows will be included in the future.

## Building

### Linux

Ensure the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `hipDNN` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/hipDNN`
- `$ cmake -S . -B build`
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/hipDNN`
- `$ make`
