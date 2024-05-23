# HIP-Basic Host Static Library Example

## Description

This example shows how to create a static library that exports hosts functions. The library may contain both `__global__` and `__device__` code as well, but in this example only `__host__` functions are exported. The resulting library may be linked with other libraries or programs, which do not necessarily need to be HIP libraries or programs. A static host library appears as a regular library, and is compatible with either hipcc or the native system's linker. When using the system linker, the libraries or applications using the static host library do need to be linked with `libamdhip64`.

### Application flow

1. The `main` function in `main.cpp` calls the library's sole exported function, `run_test`. This symbol is made visible by including the static library's header file.
2. In `run_test` in `library/library.hip`, a number of constants for the example problem are initialized.
3. A vector with input data is initialized in host memory. It is filled with an incrementing sequence starting from 0.
4. The necessary amount of device (GPU) memory is allocated and the elements of the input vectors are copied to the device memory.
5. A simple copy kernel is launched with the previously defined arguments.
6. The results are copied back to the host.
7. The previously allocated device memory is freed.
8. The results from the device are compared with the expected results on the host. An error message is printed if the results were not as expected and the function returns with an error code. If the results were as expected, the function returns 0.
9. Control flow returns to `main` in `main.cpp`, which exits the program with the value that was returned from `run_test`.

## Build Process

A HIP static host library is built the same as a regular application, except that the additional flag `--emit-static-lib` must be passed to `hipcc`. Additionally, the library should be compiled with position independent code enabled:

```shell
hipcc library/library.hip -o liblibrary.a --emit-static-lib -fPIC
```

Linking the static library with another library or object is done in the same way as a regular library:

```shell
hipcc -llibrary -Ilibrary main.cpp -o hip_static_host_library
```

Note that when linking the library using the host compiler or linker, such as `g++` or `clang++`, the `amdhip64` library should be linked with additionally:

```shell
g++ -L/opt/rocm/lib -llibrary -lamdhip64 -Ilibrary main.cpp -o hip_static_host_library
```

### CMake

Building a HIP static host library can be done using the CMake `add_library` command:

```cmake
add_library(library_name STATIC library/library.hip)
target_include_directories(library_name PUBLIC library)
```

Note that while the required compilation flags to create a library are passed to the compiler automatically by CMake, position independent code must be turned on manually:

```cmake
set_target_properties(${library_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
```

Linking with the static library is done in the same way as regular libraries. If used via `target_link_libraries`, this automatically adds the `amdhip64` dependency:

```cmake
add_executable(excutable_name main.cpp)
target_link_libraries(executable_name library_name)
```

### Visual Studio 2019

When using Visual Studio 2019 to build a HIP static host library, a separate project can be used to build the static library. This can be set up from scratch by creating a new AMD HIP C++ project, and then converting it to a library by setting `[right click project] -> Properties -> Configuration Properties -> General -> Configuration Type` to `Library`.

Linking with a HIP static host library can then be done simply by adding a reference to the corresponding project. This can be done under `[right click project] -> Add -> Reference` by checking the checkbox of the library project, and works both for AMD HIP C++ Visual Studio projects (demonstrated in [static_host_library_vs2019.vcxproj](./static_host_library_vs2019.vcxproj)) as well as regular Windows application Visual Studio projects (demonstrated in [static_host_library_msvc_vs2019.vcxproj](./static_host_library_msvc/static_host_library_msvc_vs2019.vcxproj)).

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
