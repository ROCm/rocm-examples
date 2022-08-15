*Adapt this readme - fill in the placeholders and change instructions when required. The plain text is a  template and suggestion for wording the different sections in a similar style to the `README.md`s of the other examples. Remove comments like this one when finished*
# `example-library-name` Examples

## Summary
The examples in this subdirectory showcase the functionality of the [`example-library-name`](link-to-libary) library. The examples build on [both Linux and Windows for both the ROCm (AMD GPU) and CUDA (NVIDIA GPU) backend.]

## Prerequisites
*List the prerequisites that need to be installed to compile and run the examples in the subdirectories. Provide links to the installation instructions, but the installation instructions themselves should not be part of this section.*
### Linux
*Most common prerequisites from the other examples:*
- [CMake](https://cmake.org/download/) (at least version 3.21)
- OR GNU Make - available via the distribution's package manager
- [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.2/page/Overview_of_ROCm_Installation_Methods.html) (at least version 5.x.x)
- [`example-library-name`](link-to-library): `example-library-name` package available from [repo.radeon.com](https://repo.radeon.com/rocm/). The repository is added during the standard ROCm [install procedure](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.2/page/How_to_Install_ROCm.html).


### Windows

## Building
*List the instructions to build the examples in the subdirectories. Ideally, each example builds similarly, and here the common steps can be described for both Windows and Linux. If there are example-specific build instructions, list those in the example-specific README file.*
*This directory should contain a cmake project which includes and builds all examples in the subdirectories.*

The variable `GPU_RUNTIME` can be used to set the targeted runtime. Use `HIP` to target AMD-GPUs, or `CUDA` to target NVIDIA-GPUs. When not specified, the variable defaults to `HIP`.

### Linux
Make sure that the dependencies are installed, or use the [provided Dockerfile](../../Dockerfiles/hip-libraries-rocm-ubuntu.Dockerfile) to build and run the examples in a containerized environment that has all prerequisites installed.

####  Using CMake
All examples in the `example-library-name` subdirectory can be built by a single CMake project.

- `$ cd Libraries/<example-library-name>`
- `$ cmake -S . -B build` (on ROCm) or `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (on CUDA)
- `$ cmake --build build`

#### Using Make
*Only if applicable! Not all examples have to support Make*

All examples can be built by a single invocation to Make.
- `$ cd Libraries/<example-library-name>`
- `$ make` (on ROCm) or `$ make GPU_RUNTIME=CUDA` (on CUDA)

### Windows
