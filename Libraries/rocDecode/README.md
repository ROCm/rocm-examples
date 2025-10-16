# rocDecode Examples

## Summary

The examples in this subdirectory showcase the functionality of the [rocDecode](https://github.com/ROCm/rocDecode) library. rocDecode is AMD's high-performance video decode SDK for AMD GPUs, providing hardware-accelerated video decoding capabilities. The examples demonstrate various use cases including basic video decoding, batch processing, color space conversion, and performance optimization. The examples build only on Linux for the ROCm (AMD GPU) backend.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
- Or GNU Make - available via the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 6.0)
- [rocDecode](https://github.com/ROCm/rocDecode): `rocdecode` and `rocdecode-dev` packages available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html)
- [FFMPEG](https://ffmpeg.org/about.html) development libraries:
  - On Ubuntu: `sudo apt install libavcodec-dev libavformat-dev libavutil-dev`
  - On RHEL/SLES: Install FFMPEG development packages manually or use the [rocDecode-setup.py](https://github.com/ROCm/rocDecode/blob/develop/rocDecode-setup.py) script

### Windows

Support for Windows will be included in the future.

## Building

### Linux

Ensure the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `rocDecode` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/rocDecode`
- `$ cmake -S . -B build`
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/rocDecode`
- `$ make`
