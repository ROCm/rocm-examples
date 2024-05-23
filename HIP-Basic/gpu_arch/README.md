# HIP-Basic GPU Architecture-specific Code Example

## Description

This program showcases an implementation of a simple matrix transpose kernel, which uses a different codepath depending on the target architecture.

### Application flow

1. A number of constants are defined to control the problem details and the kernel launch parameters.
2. Input matrix is set up in host memory.
3. The necessary amount of device memory is allocated and input is copied to the device.
4. The GPU transposition kernel is launched with previously defined arguments.
5. The kernel will have two different codepaths for its data movement, depending on the target architecture.
6. The transposed matrix is copied back to the host and all device memory is freed.
7. The elements of the result matrix are compared with the expected result. The result of the comparison is printed to the standard output.

## Key APIs and Concepts

This example showcases two different codepaths inside a GPU kernel, depending on the target architecture.

You may want to use architecture-specific inline assembly when compiling for a specific architecture, without losing compatibility with other architectures (see the [inline_assembly](/HIP-Basic/inline_assembly/main.hip) example).

These architecture-specific compiler definitions only exist within GPU kernels. If you would like to have GPU architecture-specific host-side code, you could query the stream/device information at runtime.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`
- `__gfx1010__`, `__gfx1011__`, `__gfx1012__`, `__gfx1030__`, `__gfx1031__`, `__gfx1100__`, `__gfx1101__`, `__gfx1102__`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
