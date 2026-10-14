# libhipcxx Examples

## Summary

The examples in this subdirectory showcase the functionality of the [libhipcxx](https://github.com/ROCm/libhipcxx) library. libhipcxx is the HIP C++ Standard Library for host and device code. It provides the C++ Standard facilities in the `hip::std::` namespace and additional extensions in the `hip::` namespace, all of which you can call from both host and device code. The `hip::` namespaces and `<hip/...>` headers are aliases for the `cuda::` namespaces and `<cuda/...>` headers.

The examples build on Linux using the ROCm platform. libhipcxx does not support Windows, so the examples do not provide Visual Studio project files, and the CMake project skips them on Windows.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
  - OR GNU Make - available via the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 7.x.x)
- [libhipcxx](https://github.com/ROCm/libhipcxx): libhipcxx must be installed. The headers are installed under `<prefix>/include/hipccl/` (with `<prefix>/include/libhipcxx/` as a fallback), and the CMake package `libhipcxx` provides the `libhipcxx::libhipcxx` target.
  - If libhipcxx is installed under `ROCM_PATH`, the examples find it automatically.
  - If libhipcxx is installed in a different location, add its installation prefix to `CMAKE_PREFIX_PATH` when you configure the CMake project.

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfile](../../Dockerfiles/ubuntu-24.04-rocm.Dockerfile) to build and run the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `libhipcxx` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/libhipcxx`
- `$ cmake -S . -B build`
  - If libhipcxx is not installed under `ROCM_PATH`, pass its installation prefix: `$ cmake -S . -B build -D CMAKE_PREFIX_PATH=<libhipcxx-prefix>`
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/libhipcxx`
- `$ make`

The Makefiles look for the libhipcxx headers in `$(ROCM_PATH)/include/hipccl`, falling back to `$(ROCM_PATH)/include/libhipcxx`. If libhipcxx is installed in a different location, pass its include directory through `CPPFLAGS`, for example `$ make CPPFLAGS="-isystem <libhipcxx-prefix>/include/hipccl"`.
