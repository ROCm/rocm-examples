# hipSOLVER Examples

## Summary

The examples in this subdirectory showcase the functionality of the [hipSOLVER](https://github.com/ROCmSoftwarePlatform/hipSOLVER) library. The examples build on both Linux and Windows for the ROCm (AMD GPU) backend.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
- OR GNU Make - available via the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 6.x.x)
- [hipSOLVER](https://github.com/ROCmSoftwarePlatform/hipSOLVER): `hipsolver` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html).

### Windows

- [Visual Studio](https://visualstudio.microsoft.com/) 2019 or 2022 with the "Desktop Development with C++" workload
- ROCm toolchain for Windows (No public release yet)
  - The Visual Studio ROCm extension needs to be installed to build with the solution files.
- [hipSOLVER](https://github.com/ROCmSoftwarePlatform/hipSOLVER)
  - Installed as part of the ROCm SDK on Windows for ROCm platform.
- [CMake](https://cmake.org/download/) (optional, to build with CMake. Requires at least version 3.21)
- [Ninja](https://ninja-build.org/) (optional, to build with CMake)

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `hipSOLVER` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/hipSOLVER`
- `$ cmake -S . -B build`
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/hipSOLVER`
- `$ make`

### Windows

#### Visual Studio

Visual Studio solution files are available for the individual examples. To build all examples for hipSOLVER open the top level solution file [ROCm-Examples-VS2017.sln](../../ROCm-Examples-VS2017.sln), [ROCm-Examples-VS2019.sln](../../ROCm-Examples-VS2019.sln) or [ROCm-Examples-VS2022.sln](../../ROCm-Examples-VS2022.sln) (for Visual Studio 2017, 2019 or 2022, respectively) and filter for hipSOLVER.

For more detailed build instructions refer to the top level [README.md](../../README.md#visual-studio).

#### CMake

All examples in the `hipSOLVER` subdirectory can either be built by a single CMake project or be built independently. For build instructions refer to the top-level [README.md](../../README.md#cmake-2).
