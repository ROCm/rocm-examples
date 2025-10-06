# MIGraphX Examples

## Summary

The examples in this subdirectory showcase the functionality of the [MIGraphX](https://github.com/ROCm/AMDMIGraphX) library. MIGraphX is AMD's graph inference engine that accelerates machine learning model inference. The examples build only on Linux for the ROCm (AMD GPU) backend and demonstrate various use cases including model parsing, quantization, custom operators, and inference workflows.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
- Or GNU Make - available via the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 7.x.x)
- [MIGraphX](https://github.com/ROCm/AMDMIGraphX): `migraphx` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html)

### Windows

MIGraphX examples are only available on Linux.

## Building

### Linux

Ensure the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `MIGraphX` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/MIGraphX`
- `$ cmake -S . -B build`
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/MIGraphX`
- `$ make`
