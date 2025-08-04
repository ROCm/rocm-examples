# HIP-Doc Compilation APIs Bitcode Example

## Description

This example demonstrates how to compile a kernel at runtime to LLVM bitcode using the HIPRTC API. For more information
on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html#bitcode).

### Application flow

1. A HIPRTC program handle is created from a string which contains HIP kernel source code.
2. The HIPRTC program is compiled to LLVM bitcode.
3. The compilation log size is queried. If the size is not `0` the actual log is obtained and printed. Any compilation
   errors will show up in this step.
4. The size of the LLVM bitcode is obtained.
5. The LLVM bitcode is loaded into a vector.
6. The program handle is destroyed.
7. Using the HIP module API, the LLVM bitcode is loaded and the contained kernel obtained.
8. Host and device data buffers are set up in the usual way.
9. The kernel is launched using the module API.
10. The result is copied to the host and validated.
11. The device memory is freed.

## Key APIs and Concepts

* `hiprtcCreateProgram` creates a HIPRTC program from a given string which contains kernel code.
* `hiprtcCompileProgram` compiles the given HIPRTC program to LLVM bitcode if the `-fgpu-rdc` flag is passed.
* `hiprtcGetProgramLogSize` returns the compilation log size. If the returned value is not `0` a warning or an error has
  occured.
* `hiprtcGetProgramLog` returns the compilation log.
* `hiprtcGetBitcodeSize` returns the size of the LLVM bitcode.
* `hiprtcGetBitcode` returns the LLVM bitcode.
* `hiprtcDestroyProgram` destroys a HIPRTC program.
* `hipModuleLoadData` builds a HIP module from a LLVM bitcode object.
* `hipModuleGetFunction` obtains a GPU kernel from a HIP module.
* `hipModuleLaunchKernel` launches a GPU kernel contained in a HIP module.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipFree`
* `hipMalloc`
* `hipMemcpy`
* `hipModuleGetFunction`
* `hipModuleLaunchKernel`
* `hipModuleLoadData`
* `hiprtcCompileProgram`
* `hiprtcCreateProgram`
* `hiprtcDestroyProgram`
* `hiprtcGetBitcode`
* `hiprtcGetBitcodeSize`
* `hiprtcGetProgramLog`
* `hiprtcGetProgramLogSize`
