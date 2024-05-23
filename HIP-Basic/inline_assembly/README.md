# HIP-Basic Inline Assembly Example

## Description

This program showcases an implementation of a simple matrix transpose kernel, which uses inline assembly and works on both AMD and NVIDIA hardware.

By using inline assembly in your kernels, you may be able to gain extra performance.
It could also enable you to use special GPU hardware features which are not available through compiler intrinsics.

For more insights, please read the following blogs by Ben Sander:
[The Art of AMDGCN Assembly: How to Bend the Machine to Your Will](https://gpuopen.com/learn/amdgcn-assembly/) &
[AMD GCN Assembly: Cross-Lane Operations](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/)

For more information:
[AMD ISA documentation for current architectures](https://gpuopen.com/amd-isa-documentation/) &
[User Guide for LLVM AMDGPU Back-end](https://llvm.org/docs/AMDGPUUsage.html)

### Application flow

1. A number of variables are defined to control the problem details and the kernel launch parameters.
2. Input matrix is set up in host memory.
3. The necessary amount of device memory is allocated and input is copied to the device.
4. The GPU transposition kernel is launched with previously defined arguments.
5. The kernel will use different inline assembly for its data movement, depending on the target platform.
6. The transposed matrix is copied back to the host and all device memory is freed.
7. The elements of the result matrix are compared with the expected result. The result of the comparison is printed to the standard output.

## Key APIs and Concepts

Using inline assembly in GPU kernels is somewhat similar to using inline assembly in host-side code. The `volatile` statement tells the compiler to not remove the assembly statement during optimizations.

```c++
asm volatile("v_mov_b32_e32 %0, %1" : "=v"(variable_0) : "v"(variable_1))
```

However, since the instruction set differs between GPU architectures, you usually want to use the appropriate GPU architecture compiler defines to support multiple architectures (see the [gpu_arch](/HIP-Basic/gpu_arch/main.hip) example for more fine-grained architecture control).

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`
- `__HIP_PLATFORM_AMD__`, `__HIP_PLATFORM_NVIDIA__`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
