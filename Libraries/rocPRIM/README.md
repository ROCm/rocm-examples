# rocPRIM Examples

## Summary
The examples in this subdirectory showcase the functionality of the [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) library. The examples build on both Linux and Windows for the ROCm (AMD GPU) backend.

## Prerequisites
### Linux
- [CMake](https://cmake.org/download/) (at least version 3.21)
- OR GNU Make - available via the distribution's package manager
- [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.1.3/page/Overview_of_ROCm_Installation_Methods.html) (at least version 5.x.x)
- [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM)
    - `rocPRIM-dev` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.1.3/page/How_to_Install_ROCm.html).

### Windows
...


## Building
### Linux
Make sure that the dependencies are installed, or use one of the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment.

#### Using CMake
All examples in the `rocPRIM` subdirectory can be built by a single CMake project.

- `$ cd Libraries/rocPRIM`
- `$ cmake -S . -B build`
- `$ cmake --build build`

#### Using Make
All examples can be built by a single invocation to Make.
- `$ cd Libraries/rocPRIM`
- `$ make`

### Windows
...
