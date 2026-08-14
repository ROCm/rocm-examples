# hipThreads Examples

## Summary

The examples in this subdirectory showcase the [hipThreads](https://github.com/ROCm/hipthreads) library. hipThreads implements C++ standard threading and synchronization primitives for AMD GPU code, enabling developers to port CPU `std::thread`-based programs to the GPU with minimal changes.

Each example is structured as a progression of steps — from a CPU-only baseline, through a minimal hipThreads drop-in port, to a fully SIMD-vectorized GPU implementation that exploits wavefront-width fiber parallelism.

The examples can be built on Linux systems for the ROCm (AMD GPU) backend. hipThreads requires ROCm 10 or later.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
- OR GNU Make - available via the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 10.x.x)
- [hipThreads](https://github.com/ROCm/hipthreads): `hipthreads` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html).
- [rocThrust](https://github.com/ROCm/rocThrust): `rocthrust` package, installed as part of the ROCm stack.
- [rocPRIM](https://github.com/ROCm/rocPRIM): `rocprim` package, installed as part of the ROCm stack.

## Building

### Linux

Make sure that the dependencies are installed, or use the [provided Dockerfiles](../../Dockerfiles/) to build and run the examples in a containerized environment that has all prerequisites installed.

#### Using CMake

All examples in the `hipThreads` subdirectory can either be built by a single CMake project or be built independently.

- `$ cd Libraries/hipThreads`
- `$ cmake -S . -B build`
- `$ cmake --build build`

#### Using Make

All examples can be built by a single invocation to Make or be built independently.

- `$ cd Libraries/hipThreads`
- `$ make`
