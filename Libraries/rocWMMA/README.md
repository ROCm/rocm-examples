# rocWMMA Examples

## Summary

The examples in this subdirectory showcase the functionality of the [rocWMMA](https://github.com/ROCm/rocWMMA) library. The examples build on Linux using the ROCm platform.

## Prerequisites

- [CMake](https://cmake.org/download/) (at least version 3.21).
- Or GNU Make - available via the distribution's package manager.
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 7.x.x).
- [rocWMMA](https://github.com/ROCm/rocWMMA): `rocwmma-dev` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html).

## Building

Ensure the dependencies are installed, or use the [provided Dockerfile](../../Dockerfiles/hip-libraries-rocm-ubuntu.Dockerfile) to build and run the examples in a containerized environment that has all prerequisites installed.

### Using CMake

All examples in the `rocWMMA` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/rocWMMA`
- `$ cmake -S . -B build`
- `$ cmake --build build`

### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/rocWMMA`
- `$ make`
