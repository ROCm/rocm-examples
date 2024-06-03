# HIP-Basic Hello World on the CUDA platform Example

## Description

This example showcases a simple HIP program that is compiled on the CUDA platform using CMake.

### Application flow

1. A kernel is launched: the function `hello_world_kernel` is executed on the device. The kernel is executed on a single thread, and prints "Hello, World!" to the console.
2. A launch error check is performed using `hipGetLastError`.
3. _Synchronization_ is performed: the host program execution halts until the kernel on the device has finished executing.

## Key APIs and Concepts

- For introduction to the programming concepts in this example, refer to the general [hello world example](../hello_world/).
- This example showcases setting up a HIP program to be compiled to the CUDA platform using CMake.
  - Since CMake (as of version 3.21) does not support compiling to CUDA in HIP language mode, CUDA language mode has to be used. Thereby the project language is specified as `CUDA`.
  - Additionally, we must "teach" CMake to compile the source file `main.hip` in CUDA language mode, because it cannot guess that from the file extension. This is done by `set_source_files_properties(main.hip PROPERTIES LANGUAGE CUDA)`.
  - The HIP "runtime" on the CUDA platform is header only. Thereby there is no need to link to a library, but the HIP include directory have to be added to the search paths. This is performed by `target_include_directories(${example_name} PRIVATE "${ROCM_ROOT}/include"`.

## Demonstrated API Calls

### HIP Runtime

- `hipGetLastError`
- `hipDeviceSynchronize`
- `__global__`

## Supported Platforms

This example is only supported on the CUDA platform.
