# HIP-Doc Lowered Names Example

## Description

This example demonstrates how to obtain the lowered (mangled) names of kernels and device variables using the HIPRTC
API. For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html#lowered-names-mangled-names).

### Application flow

1. A HIPRTC program handle is created from a string which contains HIP kernel source code.
2. The device's properties are queried in order to obtain the correct GPU architecture.
3. The HIPRTC program is compiled for the queried architecture.
4. The compilation log size is queried. If the size is not `0` the actual log is obtained and printed. Any compilation
   errors will show up in this step.
5. The size of the compiled binary code is obtained.
6. The compiled binary code is loaded into a vector.
7. The program handle is destroyed.
8. Using the HIP module API, the compiled binary code is loaded and the contained kernel obtained.
9. Host and device data buffers are set up in the usual way.
10. The kernel is launched using the module API.
11. The result is copied to the host and validated.
12. The device memory is freed.

## Key APIs and Concepts

* `hiprtcCreateProgram` creates a HIPRTC program from a given string which contains kernel code.
* `hiprtcCompileProgram` compiles the given HIPRTC program.
* `hiprtcGetProgramLogSize` returns the compilation log size. If the returned value is not `0` a warning or an error has
  occured.
* `hiprtcGetProgramLog` returns the compilation log.
* `hiprtcGetCodeSize` returns the size of the compiled binary code.
* `hiprtcGetCode` returns the compiled binary code.
* `hiprtcDestroyProgram` destroys a HIPRTC program.
* `hipModuleLoadData` builds a HIP module from a compiled binary code object.
* `hipModuleGetFunction` obtains a GPU kernel from a HIP module.
* `hipModuleLaunchKernel` launches a GPU kernel contained in a HIP module.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipFree`
* `hipGetDeviceProperties`
* `hipMalloc`
* `hipMemcpy`
* `hipModuleGetFunction`
* `hipModuleLaunchKernel`
* `hipModuleLoadData`
* `hiprtcCompileProgram`
* `hiprtcCreateProgram`
* `hiprtcDestroyProgram`
* `hiprtcGetCode`
* `hiprtcGetCodeSize`
* `hiprtcGetProgramLog`
* `hiprtcGetProgramLogSize`
