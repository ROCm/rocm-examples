# hipBLASLt Examples

## Summary

The examples in this subdirectory showcase the functionality of the [hipBLASLt](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt) library. The examples build only on Linux for the ROCm (AMD GPU) backend. Note that the examples do not perform any validation. They're intended to demonstrate how to use the API for different use cases.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21).
- Or GNU Make - available via the distribution's package manager.
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 7.x.x).
- [hipBLASLt](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt): `hipblaslt` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html).

### Windows

Support for Windows will be included in the future.

## Building

### Linux

Ensure the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `hipBLASLt` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/hipBLASLt`
- `$ cmake -S . -B build`
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/hipBLASLt`
- `$ make`
