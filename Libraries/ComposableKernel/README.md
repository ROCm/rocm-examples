# Composable Kernel Examples

## Summary

The examples in this subdirectory showcase the functionality of
[Composable Kernel](https://github.com/ROCm/composable_kernels)'s *CK Tile* subset. The examples are available only o
Linux for the ROCm (AMD GPU) backend.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
- OR GNU Make - available via the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 6.x.x)
- [Composable Kernel](https://github.com/ROCm/composable_kernel): `composablekernel` package available from
  [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm
  [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html).

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run
the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `ComposableKernel` subdirectory can either be built by a single CMake project or be built
independently.

- `$ cd Libraries/ComposableKernel`
- `$ cmake -S . -B build`
- `$ cmake --build build`
