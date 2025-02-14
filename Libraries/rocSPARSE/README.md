# rocSPARSE Examples

## Summary

The examples in this subdirectory showcase the functionality of the [rocSPARSE](https://github.com/rocmSoftwarePlatform/rocSPARSE) library. The examples build on both Linux and Windows for the ROCm (AMD GPU) backend.

## Note on asynchronous execution

All rocSPARSE library functions, unless otherwise stated, are non blocking and executed asynchronously with respect to the host. They may return before the actual computation has finished. For more information see the [rocSPARSE documentation](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/basics.html#asynchronous-execution).

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)

  - OR GNU Make - available via the distribution's package manager

- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 6.x.x)

- [rocSPARSE](https://github.com/rocmSoftwarePlatform/rocSPARSE)

  - ROCm platform: `rocsparse` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html).

### Windows

- [Visual Studio](https://visualstudio.microsoft.com/) 2019 or 2022 with the "Desktop Development with C++" workload

- ROCm toolchain for Windows (No public release yet)

  - The Visual Studio ROCm extension needs to be installed to build with the solution files.

- [rocSPARSE](https://github.com/rocmSoftwarePlatform/rocSPARSE)

  - ROCm platform: Installed as part of the ROCm SDK on Windows.

- [CMake](https://cmake.org/download/) (optional, to build with CMake. Requires at least version 3.21)

- [Ninja](https://ninja-build.org/) (optional, to build with CMake)

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment set up specifically for the example suite.

#### Using CMake

All examples in the `rocSPARSE` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/rocSPARSE`
- `$ cmake -S . -B build` (on ROCm)
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/rocSPARSE`
- `$ make`

### Windows

#### Visual Studio

Visual Studio solution files are available for the individual examples. To build all examples for rocSPARSE open the top level solution file [ROCm-Examples-VS2017.sln](../../ROCm-Examples-VS2017.sln), [ROCm-Examples-VS2019.sln](../../ROCm-Examples-VS2019.sln) or [ROCm-Examples-VS2022.sln](../../ROCm-Examples-VS2022.sln) (for Visual Studio 2017, 2019 or 2022, respectively) and filter for rocSPARSE.

For more detailed build instructions refer to the top level [README.md](../../README.md#visual-studio).

#### CMake

All examples in the `rocSPARSE` subdirectory can either be built by a single CMake project or be built independently. For build instructions refer to the top-level [README.md](../../README.md#cmake-2).
