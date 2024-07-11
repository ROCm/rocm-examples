# rocThrust Examples

## Summary

The examples in this subdirectory showcase the functionality of the [rocThrust](https://github.com/rocmSoftwarePlatform/rocThrust) library. The examples build on Linux using the ROCm platform and on Windows using the HIP on Windows platform.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
  - OR GNU Make - available via the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 6.x.x)
- [rocThrust](https://github.com/rocmSoftwarePlatform/rocThrust): `rocthrust-dev` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html).

### Windows

- [Visual Studio](https://visualstudio.microsoft.com/) 2019 or 2022 with the "Desktop Development with C++" workload
- ROCm toolchain for Windows (No public release yet)
  - The Visual Studio ROCm extension needs to be installed to build with the solution files.
- [rocThrust](https://github.com/rocmSoftwarePlatform/rocThrust): installed as part of the ROCm SDK on Windows
- [CMake](https://cmake.org/download/) (optional, to build with CMake. Requires at least version 3.21)
- [Ninja](https://ninja-build.org/) (optional, to build with CMake)

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfile](../../Dockerfiles/hip-libraries-rocm-ubuntu.Dockerfile) to build and run the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `rocThrust` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/rocThrust`
- `$ cmake -S . -B build`
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/rocThrust`
- `$ make`

### Windows

#### Visual Studio

Visual Studio solution files are available for the individual examples. To build all examples for rocThrust open the top level solution file [ROCm-Examples-VS2019.sln](../../ROCm-Examples-VS2019.sln) and filter for rocThrust.

For more detailed build instructions refer to the top level [README.md](../../README.md#visual-studio).

#### CMake

All examples in the `rocThrust` subdirectory can either be built by a single CMake project or be built independently. For build instructions refer to the top-level [README.md](../../README.md#cmake-2).
