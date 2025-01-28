# HIP-Basic Examples

## Summary

The examples in this subdirectory showcase the functionality of the HIP runtime. The examples build on Linux for the ROCm (AMD GPU) backend. Some examples additionally support Windows, some examples additionally support the CUDA (NVIDIA GPU) backend.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
- OR GNU Make - available via the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 6.x.x)

### Windows

- [Visual Studio](https://visualstudio.microsoft.com/) 2019 or 2022 with the "Desktop Development with C++" workload
- ROCm toolchain for Windows (No public release yet)
  - The Visual Studio ROCm extension needs to be installed to build with the solution files.
- [CMake](https://cmake.org/download/) (optional, to build with CMake. Requires at least version 3.21)
- [Ninja](https://ninja-build.org/) (optional, to build with CMake)

## Building

### Linux

Make sure that the dependencies are installed, or use one of the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment.

#### Using CMake

All examples in the `HIP-Basic` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd HIP-Basic`
- `$ cmake -S . -B build` (on ROCm) or `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (on CUDA, when supported)
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd HIP-Basic`
- `$ make` (on ROCm) or `$ make GPU_RUNTIME=CUDA` (on CUDA, when supported)

### Windows

Not all HIP runtime examples support building on Windows. See the README file in the directory of the example for more details.

#### Visual Studio

Visual Studio solution files are available for the individual examples. To build all supported HIP runtime examples open the top level solution file [ROCm-Examples-VS2019.sln](../../ROCm-Examples-VS2019.sln) and filter for HIP-Basic.

For more detailed build instructions refer to the top level [README.md](../../README.md#visual-studio).

#### CMake

All examples in the `HIP-Basic` subdirectory can either be built by a single CMake project or be built independently. For build instructions refer to the top-level [README.md](../../README.md#cmake-2).
