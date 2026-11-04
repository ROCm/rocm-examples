# HIP-Basic Hipify Example

## Description

The hipify example demonstrates the use of HIP utility `hifipy-perl` to port CUDA code to HIP. It converts a CUDA `.cu` source code into a portable HIP `.hip` source code that can be compiled using `amdclang++` and executed on any supported GPU (AMD or NVIDIA).

### Application flow

1. The build system (either `cmake` or `Makefile`) first converts the `main.cu` source code into a HIP portable `main.hip` source code. It uses `hipify-perl main.cu > main.hip` command to achieve the conversion.
2. `main.hip` is then compiled using `amdclang++ -x hip --offload-arch=<arch> --hip-link main.hip -o hip_hipify` to generate the executable file. `-x hip` tells the compiler to treat the input as HIP source, `--offload-arch=<arch>` selects the target GPU architecture (`--offload-arch=native` targets the GPUs present on the build machine), and `--hip-link` links against the HIP runtime.
3. The execuatable program launches a simple kernel that computes the square of each element of a vector.

## Key APIs and Concepts

`hipify-perl` is a utility that converts CUDA `.cu` source code into HIP portable code. It parses CUDA files and produces the equivalent HIP portable `.hip` source file.

## Used API surface

### HIP runtime

- `hipGetErrorString`
- `hipGetDeviceProperties`
- `hipMalloc`
