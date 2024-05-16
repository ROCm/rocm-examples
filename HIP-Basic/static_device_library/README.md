# HIP-Basic Device Static Library Example

## Description

This example shows how to create a static library that exports device functions.

### Application flow

1. A number of constants for the example problem are initialized.
2. A host vector is prepared with an increasing sequence of integers starting from 0.
3. The necessary amount of device (GPU) memory is allocated and the elements of the input vectors are copied to the device memory.
4. A simple square kernel is launched with the previously defined arguments.
5. The kernel defined in `main.hip` fetches inputs from device memory, and calls `device_square`. The `device_square` function is imported from a device static library, which is created from `library/library.hip`
6. The result from calling `device_square` is written back to device memory.
7. The results are copied back to the host.
8. The previously allocated device memory is freed.
9. The results from the device are compared with the expected results on the host. An error message is printed if the results were not as expected and the function returns with an error code.

## Build Process

Compiling a HIP static library that exports device functions must be done in two steps:

1. First, the source files that make up the library must be compiled to object files. This is done similarly to how an object file is created for a regular source file (using the `-c` flag), except that the additional option `-fgpu-rdc` must be passed:

    ```shell
    hipcc -c -fgpu-rdc -Ilibrary library/library.hip -o library.o
    ```

2. After compiling all library sources into object files, they must be manually bundled into an archive that can act as static library. `hipcc` cannot currently create this archive automatically, hence it must be created manually using `ar`:

    ```shell
    ar rcsD liblibrary.a library.o
    ```

After the static device library has been compiled, it can be linked with another HIP program or library. Linking with a static device library is done by placing it on the command line directly, and additionally requires `-fgpu-rdc`. The static library should be placed on the command line _before_ any source files. Source files that use the static library can also be compiled to object files first, in this case they also need to be compiled with `-fgpu-rdc`:

```shell
hipcc -fgpu-rdc liblibrary.a main.hip -o hip_static_device_library
```

**Note**: static device libraries _must_ be linked with `hipcc`. There is no support yet for linking such libraries with (ROCm-bundled) clang, using CMake, or using Visual Studio.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `blockDim`
- `blockIdx`
- `threadIdx`
- `__device__`
- `__global__`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
