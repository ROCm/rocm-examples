# HIP-Doc Load Module Ex Example

## Description

This example demonstrates how to load precomcpiled code objects from memory and execute the contained kernel(s). For more
information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_driver_api.html#hipmodule-api).

### Application flow

1. Two data vectors are created on the host.
2. Two data buffers are created on the device.
3. The vectors are copied to their corresponding buffers.
4. A precompiled code module is loaded from memory.
5. A kernel function is loaded from the module.
6. A buffer for kernel arguments is created. It contains both (device) data buffers.
7. A kernel launch configuration is created.
8. The kernel is launched.
9. The results are copied back to the data vectors.
10. The device buffers are freed.

## Key APIs and Concepts

* `hipModuleLoadDataEx` loads a precompiled code module from memory.
* `hipModuleGetFunction` loads a kernel from a given module.
* `hipModuleLaunchKernel` launches a kernel which was loaded from a precompiled module.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipCtxCreate`
* `hipCtxDetach`
* `hipDeviceGet`
* `hipFree`
* `hipGetErrorString`
* `hipInit`
* `hipMalloc`
* `hipMemcpyDtoH`
* `hipMemcpyHtoD`
* `hipModuleGetFunction`
* `hipModuleLaunchKernel`
* `hipModuleLoadDataEx`
