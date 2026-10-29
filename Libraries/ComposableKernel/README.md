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

As in Composable Kernel's own build, some settings depend on the GPU targets an example is built for and apply to all
of them: the examples use the OCP fp8 format when the targets include gfx950 or gfx12 (the FNUZ format otherwise),
the kernel configurations for WMMA instructions when they include gfx11 or gfx12 (the fp8 ones only when they include
gfx12 but not gfx11), and gfx950-specific features such
as tf32 GEMMs and fp8 kernel configurations when they include gfx950. Build for gfx950 and gfx12 separately from
other targets such as gfx942: in a build for both, the fp8, bf8 and tf32 results on the other targets are wrong.
