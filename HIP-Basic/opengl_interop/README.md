# HIP-Basic OpenGL Interop Example

## Description

External device resources and other handles can be shared with HIP in order to provide interoperability between different GPU APIs. This example showcases a HIP program that interacts with OpenGL: a simple HIP kernel is used to simulate a sine wave over a grid of pointers, in a buffer that is shared with OpenGL. The resulting data is then rendered to a window as a grid of triangles using OpenGL.

### Application flow

#### Initialization

1. A window is opened using the GLFW library
2. OpenGL is initialized: the window's context is made active, function pointers are loaded, debug output is enabled if possible.
3. A HIP device is picked that is OpenGL-interop capable with the current OpenGL context by using `hipGLGetDevices`.
4. OpenGL resources are initialized: a Vertex Array Object is created, buffers are created and initialized, the GLSL shader used to render the triangles is compiled.
5. A HIP stream is created on the device.
6. An OpenGL buffer is imported to HIP using `hipGraphicsGLRegisterBuffer` and `hipGraphicsMapResources`. The device pointer to this buffer is obtained with `hipGraphicsResourceGetMappedPointer`.
7. OpenGL rendering state is bound.

#### Rendering

1. The sinewave simulation kernel is launched in order to update the OpenGL shared buffer.
2. The grid is drawn to the window's framebuffer.
3. The window's framebuffer is presented to the screen.

## Dependencies

This example has additional library dependencies besides HIP:

- [GLFW](https://glfw.org). There are three options for getting this dependency satisfied:
    1. Install it through a package manager. Available for Linux, where GLFW can be installed from some of the usual package managers:

        - APT: `apt-get install libglfw3-dev`
        - Pacman: `pacman -S glfw-x11` or `pacman -S glfw-wayland`
        - DNF: `dnf install glfw-devel`

        It could also happen that the `Xxf68vm` and `Xi` libraries required when linking against Vulkan are not installed. They can be found as well on the previous package managers:
        - APT: `apt-get install libxxf86vm-dev libxi-dev`
        - Pacman: `pacman -S libxi libxxf86vm`
        - DNF: `dnf install libXi-devel libXxf86vm-devel`

    2. Build from source. GLFW supports compilation on Windows with Visual C++ (2010 and later), MinGW and MinGW-w64 and on Linux and other Unix-like systems with GCC and Clang. Please refer to the [compile guide](https://www.glfw.org/docs/latest/compile.html) for a complete guide on how to do this. Note: not only it should be built as explained in the guide, but it is additionally needed to build with the install target (`cmake --build <build-folder> --target install`).

    3. Get the pre-compiled binaries from its [download page](https://www.glfw.org/download). Available for Windows.

        Depending on the build tool used, some extra steps may be needed:

        - If using CMake, the `glfw3Config.cmake` and `glfw3Targets.cmake` files must be in a path that CMake searches by default or must be passed using `-DCMAKE_MODULE_PATH`. The official GLFW3 binaries do not ship these files on Windows, and so GLFW must either be compiled manually or obtained from [vcpkg](https://vcpkg.io/), which does ship the required cmake files.

          - If the former approach is selected, CMake will be able to find GLFW on Windows if the environment variable `GLFW3_DIR` (or the cmake option `-DCMAKE_PREFIX_PATH`) is set to (contain) the folder owning `glfw3Config.cmake` and `glfw3Targets.cmake`. For instance, if GLFW was installed in `C:\Program Files(x86)\GLFW\`, this will most surely be something like `C:\Program Files (x86)\GLFW\lib\cmake\glfw3\`.
          - If the latter, the vcpkg toolchain path should be passed to CMake using `-DCMAKE_TOOLCHAIN_FILE="/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"`.

        - If using Visual Studio, the easiest way to obtain GLFW is by installing `glfw3` from vcpkg. Alternatively, the appropriate path to the GLFW3 library and header directories can be set in `Properties->Linker->General->Additional Library Directories` and `Properties->C/C++->General->Additional Include Directories`. When using this method, the appropriate name for the GLFW library should also be updated under `Properties->C/C++->Linker->Input->Additional Dependencies`. For instance, if the path to the root folder of the Windows binaries installation was `C:\glfw-3.3.8.bin.WIN64\` and we set `GLFW_DIR` with this path, the project configuration file (`.vcxproj`) should end up containing something similar to the following:

        ```xml
        <ItemDefinitionGroup>
          <ClCompile>
            ...
            <AdditionalIncludeDirectories>$(GLFW_DIR)\include\;<other_include_directories>;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
            ...
          </ClCompile>
          <Link>
            ...
            <AdditionalDependencies>glfw3dll.lib;<other_dependencies>;%(AdditionalDependencies)</AdditionalDependencies>
            <AdditionalLibraryDirectories>$(GLFW_DIR)\lib;<other_library_directories><AdditionalLibraryDirectories>
            ...
          </Link>
        </ItemDefinitionGroup>
        ```

## Key APIs and Concepts

- `hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices, unsigned int hipDeviceCount, hipGLDeviceList deviceList)` can be used to query which HIP devices can be used to share resources with the current OpenGL context. A device returned by this function must be selected using `hipSetDevice` or a stream must be created from such a device before OpenGL interop is possible.

- `hipGraphicsGLRegisterBuffer(hipGraphicsResource_t* resource, GLuint buffer, unsigned int flags)` is used to import an OpenGL buffer into HIP. `flags` affects how the resource is used in HIP. For example:

  | flag                                   | effect                                          |
  | -------------------------------------- | ----------------------------------------------- |
  | `hipGraphicsRegisterFlagsNone`         | HIP functions may read and write to the buffer. |
  | `hipGraphicsRegisterFlagsReadOnly`     | HIP functions may only read from the buffer.    |
  | `hiPGraphicsRegisterFlagsWriteDiscard` | HIP functions may only write to the buffer.     |

- `hipGraphicsMapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream = 0)` is used to make imported OpenGL resources available to a HIP device, either the current device or a device used by a specific stream.

- `hipGraphicsResourceGetMappedPointer(void** pointer, size_t* size, hipGraphicsResource_t resource)` is used to query the device pointer that represents the memory backing the OpenGL resource. The resulting pointer may be used as any other device pointer, like those obtained from `hipMalloc`.

- `hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream = 0)` is used to unmap an imported resources from a HIP device or stream.

- `hipGraphicsUnregisterResource(hipGraphicsResource_t resource)` is used to unregister a previously imported OpenGL resource, so that it is no longer shared with HIP.

## Caveats

### Multi-GPU systems

When using OpenGL-HIP interop on multi-gpu systems, the OpenGL context must be created with the device that should be used for rendering. This is not done in this example for brevity, but is required in specific scenarios. For example, consider a multi-gpu machine with an AMD and an NVIDIA GPU: when this example is compiled for the HIP runtime, it must be launched such that the AMD GPU is used to render. A simple workaround is to launch the program from the monitor that is physically connected to the GPU to use. For multi-gpu laptops running Linux with an integrated AMD or Intel GPU and an NVIDIA dedicated gpu, the example must be launched with `__GLX_VENDOR_LIBRARY_NAME=nvidia` when compiling for NVIDIA.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`
- `blockIdx`
- `blockDim`
- `__global__`

#### Host symbols

- `hipGetDeviceProperties`
- `hipGetLastError`
- `hipGLDeviceListAll`
- `hipGLGetDevices`
- `hipGraphicsGLRegisterBuffer`
- `hipGraphicsMapResources`
- `hipGraphicsRegisterFlagsWriteDiscard`
- `hipGraphicsResourceGetMappedPointer`
- `hipGraphicsUnmapResources`
- `hipGraphicsUnregisterResource`
- `hipSetDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`
