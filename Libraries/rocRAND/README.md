# rocRAND Examples

## Summary
The examples in this subdirectory showcase the functionality of the [rocRAND](https://github.com/rocmSoftwarePlatform/rocRAND) library. The examples build on both Linux and Windows for both the ROCm (AMD GPU) and CUDA (NVIDIA GPU) backend.

## Prerequisites
### Linux
- [CMake](https://cmake.org/download/) (at least version 3.21)
- OR GNU Make - available via the distribution's package manager
- [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.1.3/page/Overview_of_ROCm_Installation_Methods.html) (at least version 5.x.x) OR the HIP Nvidia runtime (on the CUDA platform)
- [rocRAND](https://github.com/rocmSoftwarePlatform/rocRAND)
    - ROCm platform: `rocrand-dev` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.1.3/page/How_to_Install_ROCm.html).
    - CUDA platform: Install rocRAND from source: [instructions](https://github.com/rocmSoftwarePlatform/rocRAND#build-and-install).

### Windows
...

## Building
### Linux
Make sure that the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment set up specifically for the example suite.

#### Using CMake
All examples in the `rocRAND` subdirectory can be built by a single CMake project.

- `$ cd Libraries/rocRAND`
- `$ cmake -S . -B build` (on ROCm) or `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (on CUDA)
- `$ cmake --build build`

#### Using Make
All examples in the `rocRAND` subdirectory can be built by a single invocation of Make.
- `$ cd Libraries/rocRAND`
- `$ make` (on ROCm) or `$ make GPU_RUNTIME=CUDA` (on CUDA)

### Windows
...
