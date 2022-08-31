# hipCUB Examples

## Summary
The examples in this subdirectory showcase the functionality of the [hipCUB](https://github.com/ROCmSoftwarePlatform/hipCUB) library. The examples build on both Linux and Windows for both the ROCm (AMD GPU) and CUDA (NVIDIA GPU) backend.

## Prerequisites
### Linux
- [CMake](https://cmake.org/download/) (at least version 3.21)
- OR GNU Make - available via the distribution's package manager
- [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.1.3/page/Overview_of_ROCm_Installation_Methods.html) (at least version 5.x.x)
- [hipCUB](https://github.com/ROCmSoftwarePlatform/hipCUB)
    - ROCm platform: `hipCUB-dev` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.1.3/page/How_to_Install_ROCm.html).

### Windows
...

## Building
### Linux
Make sure that the dependencies are installed, or use one of the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment.

#### Using CMake
All examples in the `hipCUB` subdirectory can be built by a single CMake project.

- `$ cd Libraries/hipCUB`
- `$ cmake -S . -B build` (on ROCm) or `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (on CUDA)
- `$ cmake --build build`

#### Using Make
All examples can be built by a single invocation to Make.
- `$ cd Libraries/hipCUB`
- `$ make` (on ROCm) or `$ make GPU_RUNTIME=CUDA` (on CUDA)

### Windows
...
