# HIP-Basic Vulkan Interop Example

## Description

External device resources and other handles can be shared with HIP in order to provide interoperability between different GPU APIs. This example showcases a HIP program that interacts with the Vulkan API: A HIP kernel is used to simulate a sine wave over a grid of points, in a buffer that is shared with Vulkan. The resulting data is then rendered to a window using the Vulkan API. A set of shared semaphores is used to guarantee synchronous access to the device memory shared between HIP and Vulkan.

### Application flow

#### Initialization

1. A window is opened using the GLFW library.
2. The Vulkan API is initialized: Function pointers are loaded, the Vulkan instance is created.
3. A physical device is picked to execute the example kernel on and to render the result to the window. This physical device must be the same for HIP and for Vulkan in order to be able to share the required resources. This is done by comparing the device's UUID, which can be queried from a HIP device by querying `hipDeviceGetUuid` and from a Vulkan physical device by passing `VkPhysicalDeviceIDProperties` to `vkGetPhysicalDeviceProperties2`. If the UUIDs from a particular HIP device and Vulkan device are the same, they represent the same physical or virtual device.
4. A Vulkan logical device and related handles are initialized from the physical device handle.
5. A HIP stream is created on the same physical device.
6. A Vulkan swapchain and related handles are initialized for the window from the logical device and related handles.
7. Additional Vulkan handles required for the rendering process are initialized: A render pass, the graphics pipeline, frame buffers and other frame resources.
8. Three buffers are allocated using Vulkan: A buffer holding x- and y-coordinates for the triangle grid, a separate buffer holding a height value corresponding to each point in the triangle grid, and an index buffer that defines the triangles grid made up of the grid coordinates. The height buffer is going to be shared with HIP, and therefore it needs to be created in a way that allows it to be exported to a native memory handle. This requires passing `VkExternalMemoryBufferCreateInfoKHR` to `vkCreateBuffer` when creating a buffer, with `VkExternalMemoryBufferCreateInfoKHR::handleTypes` initialized to the appropriate type for the native platform. Additionally, this requires setting the same value on `VkExportMemoryAllocateInfoKHR`, which must be passed to `vkAllocateMemory` when allocating memory for a buffer that is to be exported.
9. The x- and y-coordinates buffer and the index buffer are initialized with their contents. These buffers do not change during the duration of the program.
10. A HIP external memory handle is created from the Vulkan height buffer memory handle. This is done by first exporting the Vulkan buffer to a platform-native handle using `VkGetMemoryFd` or `VkGetMemoryWin32Handle` depending on the platform, and then importing that handle to HIP using `hipImportExternalMemory`.
11. A pointer to the device memory of the height buffer is obtained from the HIP external memory handle using `hipExternalMemoryGetMappedBuffer`.
12. Two semaphores used to synchronize memory accesses between HIP and Vulkan are initialized: The first synchronizes the access from when the buffer was used to render the previous frame in Vulkan to when the HIP kernel is invoked, , and the second synchronizes the access from when the HIP kernel is finished to when Vulkan can use the buffer to render the next frame. Similar to buffers, these must be created in a way that allows them to be exported to a platform-native semaphore handle, so that they may later be imported as HIP external semaphore. This is done by passing `VkExportSemaphoreCreateInfoKHR` to `vkCreateSemaphore`, of which `handleTypes` must again be initialized to the appropriate platform-dependent handle type.
13. The Vulkan semaphores are converted to HIP external semaphores. This is done by first exporting a Vulkan semaphore handle to a native semaphore handle, either by `vkGetSemaphoreFdKHR` or `vkGetSemaphoreWin32HandleKHR` depending on the target platform. The resulting handle is passed to `hipImportExternalSemaphore` to obtain the HIP semaphore handle.

#### Rendering

A frame is rendered as follows:

1. The frame resources for the current frame in the frame pipeline are fetched from memory.
2. The next image index is acquired from the swapchain.
3. The command pool associated to the current frame is reset and the associated command buffer is initialized.
4. `hipWaitExternalSemaphoresAsync` is used to ensure that Vulkan has finished rendering the previous frame before the HIP kernel is invoked. Note that this function is not required on the first frame.
5. The HIP kernel is invoked.
6. `hipSignalExternalSemaphoresAsync` is used to signal Vulkan that HIP is now finished with the buffer and that Vulkan can proceed with rendering.
7. The Vulkan rendering commands are recorded to the current frame's command buffer.
8. The command buffer is submitted to the Vulkan graphics queue. The semaphore that synchronizes the HIP kernel invocation with the Vulkan rending commands is passed to `VkSubmitInfo::pWaitSemaphores` to make Vulkan wait on the semaphore signal before proceeding with rendering. As a small optimization, the corresponding element in `VkSubmitInfo::pWaitDstStageMask` is set to `VK_PIPELINE_STAGE_VERTEX_INPUT_BIT`. The height buffer is only needed at the vertex input stage, and this way the prior stages can already be executed by Vulkan even if the semaphore is not signaled yet. The semaphore that synchronizes between rendering the previous frame and running the HIP kernel for the next frame is passed to `vkSubmitInfo::pSignalSemaphores`, so that Vulkan signals it when the frame is finished with rendering.
9. The swapchain is asked to present the current frame to the screen.

## Key APIs and Concepts

To share memory allocated by Vulkan with HIP, the `VkDeviceMemory` must be created by passing the `VkExportMemoryAllocateInfoKHR` structure to `vkAllocateDeviceMemory`. This structure needs the appropriate `handleTypes` set to a type that can be shared with HIP for the current platform; `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR` for Linux and `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR` or `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR` for Windows. Any Vulkan buffer that is to be associated with this device memory must similarly be created by passing `VkExternalMemoryBufferCreateInfoKHR` to `vkCreateBuffer`, of which the `handleTypes` member must be initialized to the same value. The `VkDeviceMemory` handle can then be exported to a native file descriptor or `HANDLE` using `vkGetMemoryFdKHR` or `vkGetMemoryWin32HandleKHR` respectively on Linux and Windows. A `hipExternalMemory_t` can then be imported from a native handle through `hipImportExternalMemory`. This function must be passed an instance of `hipExternalmemoryHandleDesc`, of which `type` is initialized with a handle type compatible with the Vulkan `handleTypes`. This mapping is as follows:

| Vulkan memory handle type                                 | HIP memory handle type                      |
| --------------------------------------------------------- | ------------------------------------------- |
| `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR`        | `hipExternalMemoryHandleTypeOpaqueFd`       |
| `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR`     | `hipExternalMemoryHandleTypeOpaqueWin32`    |
| `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR` | `hipExternalMemoryHandleTypeOpaqueWin32Kmt` |

To actually use this external memory handle in HIP the corresponding HIP device memory pointer should first be obtained. This can be done with the `hipExternalMemoryGetMappedBuffer` function.

Sharing semaphores follows a similar process: The `VkSemaphore` must be created by passing `VkExportSemaphoreCreateInfoKHR`, of which `handleTypes` must be initialized to `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR` for Linux, or `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR` or `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR` for Windows. The `VkSemaphore` handle can then be exported to a native Linux file descriptor or Windows `HANDLE` using `vkGetSemaphoreFdKHR` or `vkGetSemaphoreWin32HandleKHR` on Linux and Windows respectively. The `hipExternalSemaphore_t` can then be created using `hipImportExternalSemaphore`. It must be passed an instance of `hipExternalSemaphoreHandleDesc`, of which `type` is again initialized with a compatible HIP-version of the Vulkan `handleTypes`. This mapping is as follows:

| Vulkan semaphore handle type                                 | HIP semaphore handle type                      |
| ------------------------------------------------------------ | ---------------------------------------------- |
| `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR`        | `hipExternalSemaphoreHandleTypeOpaqueFd`       |
| `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR`     | `hipExternalSemaphoreHandleTypeOpaqueWin32`    |
| `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR` | `hipExternalSemaphoreHandleTypeOpaqueWin32Kmt` |

To wait on a shared semaphore in HIP, `hipWaitExternalSemaphoresAsync` should be used. This must be passed a number of `hipExternalSemaphoreWaitParams` structures, each corresponding to a semaphore with the same index. When using timeline semaphores, its `fence.value` member can be used to specify which timeline semaphore value to wait on.

To signal a shared semaphore in HIP, the `hipSignalExternalSemaphoresAsync` function can be used. This must be passed a number of `hipExternalSemaphoreSignalParams` structures, each corresponding to a semaphore with the same index. When using timeline semaphores, its `fence.value` member should be set to specify the value to which the semaphore should be set.

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

- Vulkan headers. On Linux, the vulkan headers can be directly obtained from some package managers:

  - Linux

    - APT: `apt-get install -y libvulkan-dev`
    - Pacman: `pacman -S vulkan-headers vulkan-icd-loader`
    - DNF: `dnf install vulkan-headers vulkan-icd-loader`

  But they may be as well obtained by installing the [LunarG Vulkan SDK](https://vulkan.lunarg.com/). CMake will be able to find the SDK using the `VULKAN_SDK` environment variable, which is set by default using the SDK activation script.

  On Windows, on the other hand, the headers can only be obtained from the [LunarG Vulkan SDK](https://vulkan.lunarg.com/). Contrary to Unix-based OSs, the `VULKAN_SDK` environment variable is not automatically provided on Windows, and so it should be set to the appropriate path before invoking CMake.

  Note that `libvulkan` is _not_ required, as the example loads function pointers dynamically.

- Validation layers. The `VK_LAYER_KHRONOS_validation` layer is active by default to perform general checks on Vulkan, thus the [Khronos' Vulkan Validation Layers (VVL)](https://github.com/KhronosGroup/Vulkan-ValidationLayers/tree/main#vulkan-validation-layers-vvl) will need to be installed on the system if such checks are desirable. It can be either installed from a package manager (on Linux), built and configured from source or installed as part of the [LunarG Vulkan SDK](https://vulkan.lunarg.com/).

  Package managers offering the validation layers package include:

  - APT: `apt install vulkan-validationlayers-dev`
  - Pacman: `pacman -S vulkan-validation-layers`. Note that with pacman both the validation layers and headers (among others) can be also installed with `pacman -S vulkan-devel`.
  - DNF: `dnf install vulkan-validation-layers`

  For the second approach, build instructions are provided on [Khronos Vulkan-ValidationLayers repository](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/BUILD.md) and Vulkan's [Layers Overwiew and Configuration](https://vulkan.lunarg.com/doc/view/latest/windows/layer_configuration.html) document offers several approaches for its configuration.

- `glslangValidator`. It is used in the example as a shader validation tool. It may be installed via package manager (`sudo apt install glslang-tools`), by [building manually from source](https://github.com/KhronosGroup/glslang#building-cmake), by downloading the binaries for the corresponding platform directly from the [main-tot](https://github.com/KhronosGroup/glslang/releases/tag/main-tot) release on GitHub or installed as part of the [LunarG Vulkan SDK](https://vulkan.lunarg.com/).

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`

#### Host symbols

- `hipComputeModeProhibited`
- `hipCUDAErrorTohipError`
- `hipDestroyExternalMemory`
- `hipDestroyExternalSemaphore`
- `hipDeviceGetUuid`
- `hipExternalMemoryBufferDesc`
- `hipExternalMemoryGetMappedBuffer`
- `hipExternalMemoryHandleDesc`
- `hipExternalMemoryHandleType`
- `hipExternalMemoryHandleTypeOpaqueFd`
- `hipExternalMemoryHandleTypeOpaqueWin32`
- `hipExternalMemoryHandleTypeOpaqueWin32Kmt`
- `hipExternalSemaphoreHandleDesc`
- `hipExternalSemaphoreHandleType`
- `hipExternalSemaphoreHandleTypeOpaqueFd`
- `hipExternalSemaphoreHandleTypeOpaqueWin32`
- `hipExternalSemaphoreHandleTypeOpaqueWin32Kmt`
- `hipExternalSemaphoreSignalParams`
- `hipExternalSemaphoreWaitParams`
- `hipGetDeviceCount`
- `hipGetDeviceProperties`
- `hipGetLastError`
- `hipImportExternalMemory`
- `hipImportExternalSemaphore`
- `hipSetDevice`
- `hipSignalExternalSemaphoresAsync`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`
- `hipWaitExternalSemaphoresAsync`
