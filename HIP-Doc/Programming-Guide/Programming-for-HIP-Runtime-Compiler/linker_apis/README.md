# HIP-Doc Linker APIs Example

## Description

This example demonstrates how to link at runtime a LLVM bitcode object (which is stored in memory) using the HIPRTC API.
For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html#linker-apis).

### Application flow

1. A HIPRTC program handle is created from a string which contains HIP kernel source code.
2. The HIPRTC program is compiled to LLVM bitcode.
3. The compilation log size is queried. If the size is not `0` the actual log is obtained and printed. Any compilation
   errors will show up in this step.
4. The size of the LLVM bitcode is obtained.
5. The LLVM bitcode is loaded into a vector.
6. The program handle is destroyed.
7. A HIPRTC link instance is created.
8. The LLVM bitcode is added to the link instance.
9. The link is completed and a binary object is produced.
10. Using the HIP module API, the binary object is loaded and the contained kernel obtained.
11. The link instance is destroyed.
12. Host and device data buffers are set up in the usual way.
13. The kernel is launched using the module API.
14. The result is copied to the host and validated.
15. The device memory is freed.

## Key APIs and Concepts

* `hiprtcCreateProgram` creates a HIPRTC program from a given string which contains kernel code.
* `hiprtcCompileProgram` compiles the given HIPRTC program to LLVM bitcode if the `-fgpu-rdc` flag is passed.
* `hiprtcGetProgramLogSize` returns the compilation log size. If the returned value is not `0` a warning or an error has
  occured.
* `hiprtcGetProgramLog` returns the compilation log.
* `hiprtcGetBitcodeSize` returns the size of the LLVM bitcode.
* `hiprtcGetBitcode` returns the LLVM bitcode.
* `hiprtcDestroyProgram` destroys a HIPRTC program.
* `hiprtcLinkCreate` creates a HIPRTC link instance.
* `hiprtcLinkAddData` adds LLVM bitcode (which is stored in memory) to the link instance.
* `hiprtcLinkComplete` completes the link and produces a binary object.
* `hipModuleLoadData` builds a HIP module from a binary object.
* `hiprtcLinkDestroy` destroys a link instance.
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
* `hiprtcLinkAddData`
* `hiprtcLinkComplete`
* `hiprtcLinkCreate`
* `hiprtcLinkDestroy`
