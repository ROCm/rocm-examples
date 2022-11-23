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
- [GLFW3](https://glfw.org). GLFW can be installed either through the package manager, or can be obtained from its home page. If using CMake, the `glfw3Config.cmake` file must be in a path that CMake searches by default or must be passed using `-DCMAKE_MODULE_PATH`.
The official GLFW3 binaries do not ship this file on Windows, and so GLFW3 must either be compiled manually. CMake will be able to find GLFW on Windows if it is installed in `C:\Program Files(x86)\glfw\`. Alternatively, GLFW can be obtained from [vcpkg](https://vcpkg.io/), which does ship the required cmake files. In this case, the vcpkg toolchain path should be passed to CMake using `-DCMAKE_TOOLCHAIN_FILE="/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"`.
If using Visual Studio, the easiest way to obtain GLFW is by installing glfw3 from vcpkg. Alternatively, the appropriate path to the GLFW3 library and header directories can be set in Properties->Linker->General->Additional Library Directories and Properties->C/C++->General->Additional Include Directories. When using this method, the appropriate name for the glfw library should also be updated under Properties->C/C++->Linker->Input->Additional Dependencies.

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
- `hipLaunchKernelGGL`
- `hipSetDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`
