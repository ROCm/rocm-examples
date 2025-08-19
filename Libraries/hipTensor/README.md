# hipTensor Examples

## Summary

The examples in this subdirectory showcase the functionality of the [hipTensor](https://github.com/ROCm/hipTensor) library. The examples build only on Linux for the ROCm (AMD GPU) backend.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21).
- Or GNU Make - available via the distribution's package manager.
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 7.x.x).
- [hipTensor](https://github.com/ROCm/hipTensor): `hiptensor` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html).

### Windows

Support for Windows will be included in the future.

## Building

### Linux

Ensure the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `hipTensor` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/hipTensor`
- `$ cmake -S . -B build`
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/hipTensor`
- `$ make`
