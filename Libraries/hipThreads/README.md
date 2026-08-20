# hipThreads Examples

## Summary

The examples in this subdirectory showcase the [hipThreads](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipthreads) library. hipThreads implements C++ standard threading and synchronization primitives for AMD GPU code, enabling developers to port CPU `std::thread`-based programs to the GPU with minimal changes.

Each example is structured as a progression of steps — from a CPU-only baseline, through a minimal hipThreads drop-in port, to a fully SIMD-vectorized GPU implementation that exploits wavefront-width fiber parallelism.

The examples can be built on Linux systems for the ROCm (AMD GPU) backend.

> **Note:** hipThreads has not yet shipped in a stable ROCm release. Development builds are
> available through the [TheRock](https://github.com/ROCm/TheRock) project.

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
- OR GNU Make - available via the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 10.x.x)
- [hipThreads](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipthreads): not yet available as a package from [repo.radeon.com](https://repo.radeon.com/rocm/); build from source via [TheRock](https://github.com/ROCm/TheRock).
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
