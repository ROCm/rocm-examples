# ROCm Examples

This repository is a collection of examples to enable new users to start using
ROCm, as well as provide more advanced examples for experienced users.

The examples are structured in several categories:

- [HIP-Basic](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/) showcases some basic functionality without any additional dependencies
- [HIP-Doc](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/) contains the example codes that are shown in the [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [Libraries](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/) contains examples for ROCm-libraries, that provide higher-level functionality
- [Applications](https://github.com/ROCm/rocm-examples/tree/amd-staging/Applications/) showcases some common applications, using HIP to accelerate them

- [AI](https://github.com/ROCm/rocm-examples/tree/amd-staging/AI/) contains instructions on how to use ROCm for AI
- [Tutorials](https://github.com/ROCm/rocm-examples/tree/amd-staging/Tutorials/) contains the code accompanying the HIP Tutorials that can be found in [the HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/saxpy.html).

For a full overview over the examples see the section [repository contents](#repository-contents).

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
- A number of examples also support building via  GNU Make - available through the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 7.x.x)
- For example-specific prerequisites, see the example subdirectories.

### Windows

- [Visual Studio](https://visualstudio.microsoft.com/) 2019 or 2022 with the "Desktop Development with C++" workload
- [HIP SDK for Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html)
  - The Visual Studio ROCm extension needs to be installed to build with the solution files.
- [CMake](https://cmake.org/download/) (optional, to build with CMake. Requires at least version 3.21)
- [Ninja](https://ninja-build.org/) (optional, to build with CMake)
- [Perl](https://www.perl.org/get.html) (for hipify related scripts)

## Building the example suite

### Linux

These instructions assume that the prerequisites for every example are installed on the system.

#### CMake

See [CMake build options](#cmake-build-options) for an overview of build options.

- `$ git clone https://github.com/ROCm/rocm-examples.git`
- `$ cd rocm-examples`
- `$ cmake -S . -B build` (on ROCm) or `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (on CUDA)
- `$ cmake --build build`
- `$ cmake --install build --prefix install`

#### Make

Beware that only a subset of the examples support building via Make.

- `$ git clone https://github.com/ROCm/rocm-examples.git`
- `$ cd rocm-examples`
- `$ make` (on ROCm) or `$ make GPU_RUNTIME=CUDA` (on CUDA)

### Linux with Docker

Alternatively, instead of installing the prerequisites on the system, the [Dockerfiles](https://github.com/ROCm/rocm-examples/tree/amd-staging/Dockerfiles/) in this repository can be used to build images that provide all required prerequisites. Note, that the ROCm kernel GPU driver still needs to be installed on the host system.

The following instructions showcase building the Docker image and full example suite inside the container using CMake:

- `$ git clone https://github.com/ROCm/rocm-examples.git`
- `$ cd rocm-examples/Dockerfiles`
- `$ docker build . -t rocm-examples -f hip-libraries-rocm-ubuntu.Dockerfile --build-arg GID="$(getent group render | cut -d':' -f 3)"` (on ROCm) or `$ docker build . -t rocm-examples -f hip-libraries-cuda-ubuntu.Dockerfile` (on CUDA)
- `$ docker run -it --device /dev/kfd --device /dev/dri rocm-examples bash` (on ROCm) or `$ docker run -it --gpus=all rocm-examples bash` (on CUDA)
- `# git clone https://github.com/ROCm/rocm-examples.git`
- `# cd rocm-examples`
- `# cmake -S . -B build` (on ROCm) or `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (on CUDA)
- `# cmake --build build`

The built executables can be found and run in the `build` directory:

- `# ./build/Libraries/rocRAND/simple_distributions_cpp/simple_distributions_cpp`

### Windows

#### Visual Studio

The repository has Visual Studio project files for all examples and individually for each example.

- Project files for Visual Studio are named as the example with `_vs<Visual Studio Version>` suffix added e.g. `device_sum_vs2019.sln` for the device sum example.
- The project files can be built from Visual Studio or from the command line using MSBuild.
  - Use the build solution command in Visual Studio to build.
  - To build from the command line execute `C:\Program Files (x86)\Microsoft Visual Studio\<Visual Studio Version>\<Edition>\MSBuild\Current\Bin\MSBuild.exe <path to project folder>`.
    - To build in Release mode pass the `/p:Configuration=Release` option to MSBuild.
    - The executables will be created in a subfolder named "Debug" or "Release" inside the project folder.
- The HIP specific project settings like the GPU architectures targeted can be set on the `General [AMD HIP C++]` tab of project properties.
- The top level solution files come in two flavors: `ROCm-Examples-VS<Visual Studio Verson>.sln` and `ROCm-Examples-Portable-VS<Visual Studio Version>.sln`. The former contains all examples, while the latter contains the examples that support both ROCm and CUDA.

#### CMake

First, clone the repository and go to the source directory.

```shell
git clone https://github.com/ROCm/rocm-examples.git
cd rocm-examples
```

There are two ways to build the project using CMake: with the Visual Studio Developer Command Prompt (recommended) or with a standard Command Prompt. See [CMake build options](#cmake-build-options) for an overview of build options.

##### Visual Studio Developer Command Prompt

Select Start, search for "x64 Native Tools Command Prompt for VS 2019", and the resulting Command Prompt. Ninja must be selected as generator, and Clang as C++ compiler.

```shell
cmake -S . -B build -G Ninja -D CMAKE_CXX_COMPILER=clang
cmake --build build
```

##### Standard Command Prompt

Run the standard Command Prompt. When using the standard Command Prompt to build the project, the Resource Compiler (RC) path must be specified. The RC is a tool used to build Windows-based applications, its default path is `C:/Program Files (x86)/Windows Kits/10/bin/<Windows version>/x64/rc.exe`. Finally, the generator must be set to Ninja.

```shell
cmake -S . -B build -G Ninja -D CMAKE_RC_COMPILER="<path to rc compiler>"
cmake --build build
```

### CMake build options

The following options are available when building with CMake.

| Option                     | Relevant to | Default value    | Description                                                                                             |
|:---------------------------|:------------|:-----------------|:--------------------------------------------------------------------------------------------------------|
| `GPU_RUNTIME`              | HIP / CUDA  | `"HIP"`          | GPU runtime to compile for. Set to `"CUDA"` to compile for NVIDIA GPUs and to `"HIP"` for AMD GPUs.     |
| `CMAKE_HIP_ARCHITECTURES`  | HIP         | Compiler default | HIP device architectures to target, e.g. `"gfx908;gfx1030"` to target architectures gfx908 and gfx1030. |
| `CMAKE_CUDA_ARCHITECTURES` | CUDA        | Compiler default | CUDA architecture to compile for e.g. `"50;72"` to target compute capibility 50 and 72.                 |

## Repository Contents

- [AI](https://github.com/ROCm/rocm-examples/tree/amd-staging/AI/MIGraphX/Quantization) Showcases the functionality for executing quantized models using Torch-MIGraphX.
- [Applications](https://github.com/ROCm/rocm-examples/tree/amd-staging/Applications/) groups a number of examples ... .
  - [bitonic_sort](https://github.com/ROCm/rocm-examples/tree/amd-staging/Applications/bitonic_sort/): Showcases how to order an array of $n$ elements using a GPU implementation of the bitonic sort.
  - [convolution](https://github.com/ROCm/rocm-examples/tree/amd-staging/Applications/convolution/): A simple GPU implementation for the calculation of discrete convolutions.
  - [floyd_warshall](https://github.com/ROCm/rocm-examples/tree/amd-staging/Applications/floyd_warshall/): Showcases a GPU implementation of the Floyd-Warshall algorithm for finding shortest paths in certain types of graphs.
  - [histogram](https://github.com/ROCm/rocm-examples/tree/amd-staging/Applications/histogram/): Histogram over a byte array with memory bank optimization.
  - [monte_carlo_pi](https://github.com/ROCm/rocm-examples/tree/amd-staging/Applications/monte_carlo_pi/): Monte Carlo estimation of $\pi$ using hipRAND for random number generation and hipCUB for evaluation.
  - [prefix_sum](https://github.com/ROCm/rocm-examples/tree/amd-staging/Applications/prefix_sum/): Showcases a GPU implementation of a prefix sum with a 2-kernel scan algorithm.
- [Common](https://github.com/ROCm/rocm-examples/tree/amd-staging/Common/) contains common utility functionality shared between the examples.
- [HIP-Basic](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/) hosts self-contained recipes showcasing HIP runtime functionality.
  - [assembly_to_executable](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/assembly_to_executable): Program and accompanying build systems that show how to manually compile and link a HIP application from host and device code.
  - [bandwidth](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/bandwidth): Program that measures memory bandwidth from host to device, device to host, and device to device.
  - [bit_extract](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/bit_extract): Program that showcases how to use HIP built-in bit extract.
  - [device_globals](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/device_globals): Show cases how to set global variables on the device from the host.
  - [device_query](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/device_query): Program that showcases how properties from the device may be queried.
  - [dynamic_shared](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/dynamic_shared): Program that showcases how to use dynamic shared memory with the help of a simple matrix transpose kernel.
  - [events](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/events/): Measuring execution time and synchronizing with HIP events.
  - [gpu_arch](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/gpu_arch/): Program that showcases how to implement GPU architecture-specific code.
  - [hello_world](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/hello_world): Simple program that showcases launching kernels and printing from the device.
  - [hello_world_cuda](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/hello_world_cuda): Simple HIP program that showcases setting up CMake to target the CUDA platform.
  - [hipify](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/hipify): Simple program and build definitions that showcase automatically converting a CUDA `.cu` source into portable HIP `.hip` source.
  - [llvm_ir_to_executable](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/llvm_ir_to_executable): Shows how to create a HIP executable from LLVM IR.
  - [inline_assembly](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/inline_assembly/): Program that showcases how to use inline assembly in a portable manner.
  - [matrix_multiplication](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/matrix_multiplication/): Multiply two dynamically sized matrices utilizing shared memory.
  - [module_api](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/module_api/): Shows how to load and execute a HIP module in runtime.
  - [moving_average](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/moving_average/): Simple program that demonstrates parallel computation of a moving average of one-dimensional data.
  - [multi_gpu_data_transfer](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/multi_gpu_data_transfer/): Performs two matrix transposes on two different devices (one on each) to showcase how to use peer-to-peer communication among devices.
  - [occupancy](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/occupancy/): Shows how to find optimal configuration parameters for a kernel launch with maximum occupancy.
  - [opengl_interop](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/opengl_interop): Showcases how to share resources and computation between HIP and OpenGL.
  - [runtime_compilation](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/runtime_compilation/): Simple program that showcases how to use HIP runtime compilation (hipRTC) to compile a kernel and launch it on a device.
  - [saxpy](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/saxpy/): Implements the $y_i=ax_i+y_i$ kernel and explains basic HIP functionality.
  - [shared_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/shared_memory/): Showcases how to use static shared memory by implementing a simple matrix transpose kernel.
  - [static_device_library](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/static_device_library): Shows how to create a static library containing device functions, and how to link it with an executable.
  - [static_host_library](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/static_host_library): Shows how to create a static library containing HIP host functions, and how to link it with an executable.
  - [streams](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/streams/): Program that showcases usage of multiple streams each with their own tasks.
  - [texture_management](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/texture_management/): Shows the usage of texture memory.
  - [vulkan_interop](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/vulkan_interop): Showcases how to share resources and computation between HIP and Vulkan via buffer memory.
  - [vulkan_interop_mipmap](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/vulkan_interop): Showcases how to share resources and computation between HIP and Vulkan via mipmapped memory (Only available in Windows).
  - [warp_shuffle](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/warp_shuffle/): Uses a simple matrix transpose kernel to showcase how to use warp shuffle operations.
- [HIP-Doc](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc) hosts the [HIP documentation's](https://rocm.docs.amd.com/projects/HIP/en/latest/) example codes. These are mainly intended for CI purposes but also serve as standalone examples.
  - [Programming-Guide](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide) contains the examples from the HIP documentation's Programming Guide section.
    - [HIP-C++-Language-Extensions](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions) contains the examples from the [HIP C++ language extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html) page.
      - [calling_global_functions](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions/calling_global_functions): Shows how to call `__global__` functions (kernels).
      - [extern_shared_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions/extern_shared_memory): Shows how to dynamically allocate `__shared__` memory.
      - [launch_bounds](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions/launch_bounds): Shows how to specify launch bounds for a kernel.
      - [set_constant_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions/set_constant_memory): Shows how to initialize `__constant__` memory.
      - [template_warp_size_reduction](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions/template_warp_size_reduction): Shows how to perform a reduction while relying on the warp size as a compile-time constant.
      - [timer](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions/timer): Shows how to read the device-side timer.
      - [warp_size_reduction](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions/warp_size_reduction): Shows how to perform a reduction while relying on the warp size as an early-folded constant.
    - [Porting-CUDA-code-to-HIP](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP) contains the examples from the [Porting NVIDIA CUDA code to HIP guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html) page.
      - [address_retrieval](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/address_retrieval): Shows how to obtain the address of a HIP runtime function.
      - [device_code_feature_identification](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/device_code_feature_identification): Shows how to query the device's compute features in device code.
      - [host_code_feature_identification](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/host_code_feature_identification): Shows how to query the device's compute features in host code.
      - [identifying_compilation_target_platform](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/identifying_compilation_target_platform): Shows how to distinguish between AMD and NVIDIA target platforms in code.
      - [identifying_host_device_compilation_pass](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/identifying_host_device_compilation_pass): Shows how to disinguish between host and device compilation passes in code.
      - [load_module](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/load_module): Shows how to load precompiled code objects from disk and execute the contained kernel(s).
      - [load_module_ex](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/load_module_ex): Shows how to load precompiled code objects from memory and execute the contained kernel(s).
      - [load_module_ex_cuda](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/load_module_ex_cuda): Shows how to load CUDA PTX objects from disk and execute the contained kernel(s).
      - [per_thread_default_stream](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/per_thread_default_stream): Shows how to manage streams on a per-thread basis.
      - [pointer_memory_type](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/pointer_memory_type): Shows how to query a pointer's memory type.
    - [Introduction-to-the-HIP-Programming-Model](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Introduction-to-the-HIP-Programming-Model) contains the examples from the [Introduction to the HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html) page.
      - [add_kernel](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Introduction-to-the-HIP-Programming-Model/add_kernel): Shows how to perform a vector addition with a GPU kernel.
    - [Programming-for-HIP-Runtime-Compiler](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler) contains the examples from the [Programming for HIP runtime compiler (RTC)](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html) page.
      - [compilation_apis](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/compilation_apis): Shows how to compile a kernel at runtime using the HIPRTC API.
      - [linker_apis](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/linker_apis): Shows how to link at runtime a LLVM bitcode object (which is stored in memory) using the HIPRTC API.
      - [linker_apis_file](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/linker_apis_file): Shows how to link at runtime a LLVM bitcode object (which is stored in a file on disk) using the HIPRTC API.
      - [linker_apis_options](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/linker_apis_options): Shows how to link at runtime a LLVM bitcode object (which is stored in memory) using the HIPRTC API. During the link stage an array of linker options is passed to the runtime linker.
      - [lowered_names](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/lowered_names): Shows how to obtain the lowered (mangled) names of kernels and device variables using the HIPRTC API.
      - [rtc_error_handling](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/rtc_error_handling): Shows how to check the HIPRTC API calls for errors.
    - [Using-HIP-Runtime-API](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API) contains the examples from the [Using HIP runtime API](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api.html) subsection.
      - [Asynchronous-Concurrent-Execution](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution) contains the examples from the [Asynchronous concurrent execution](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html) page.
        - [async_kernel_execution](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution/async_kernel_execution): Shows how to execute HIP operations and kernels asynchronously with regard to the host.
        - [event_based_synchronization](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution/event_based_synchronization): Shows how to execute HIP operations and kernels asynchronously with regard to the host and how to synchronize the host and the device by using HIP events.
        - [sequential_kernel_execution](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution/sequential_kernel_execution): Shows how to execute HIP operations and kernels sequentially.
      - [Call-Stack](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Call-Stack) contains the examples from the [Call stack](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/call_stack.html) page.
        - [call_stack_management](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Call-Stack/call_stack_management): Shows how to adjust the device's call stack size.
        - [device_recursion](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Call-Stack/device_recursion): Shows how to hit the device's stack limit on purpose.
      - [Error-Handling](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Error-Handling) contains the examples from the [Error handling](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/error_handling.html) page.
        - [error_handling](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Error-Handling/error_handling): Shows how to handle HIP runtime errors without creating too much code overhead.
      - [HIP-Graphs](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/HIP-Graphs) contains the examples from the [HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html) page.
        - [graph_capture](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/HIP-Graphs/graph_capture): Shows how to capture HIP streams with the HIP graph API.
        - [graph_creation](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/HIP-Graphs/graph_creation): Shows how to explicitly create HIP graphs.
      - [Initialization](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Initialization) contains the examples from the [Initialization](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/initialization.html) page.
        - [simple_device_query](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Initialization/simple_device_query): Shows how the number of HIP-capable devices in the system can be determined, as well as how properties from the device may be queried.
      - [Memory-Management](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management) contains the examples from the [Memory management](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html) subsubsection.
        - [Device-Memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory) contains the examples from the [Device memory](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/device_memory.html) page.
          - [constant_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/constant_memory): Shows how to transfer bytes between the host and the device's constant memory space.
          - [dynamic_shared_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/dynamic_shared_memory): Shows how to dynamically allocate shared memory on the host.
          - [explicit_copy](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/explicit_copy): Shows how to transfer bytes between the host and the device's global memory space.
          - [kernel_memory_allocation](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/kernel_memory_allocation): Shows how to allocate global device memory inside a kernel.
          - [static_shared_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/static_shared_memory): Shows how to statically allocate shared memory inside a kernel.
        - [Host-Memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Host-Memory) contains the examples from the [Host memory](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/host_memory.html) page.
          - [pageable_host_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Host-Memory/pageable_host_memory): Shows how to allocate pageable memory on the host and transfer its contents to the device.
          - [pinned_host_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Host-Memory/pinned_host_memory): Shows how to allocate pinned memory on the host and transfer its contents to the device.
        - [SOMA](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA) contains the examples from the [Stream Ordered Memory Allocator](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html) page.
          - [memory_pool](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool): Shows how to use the stream ordered memory allocation (SOMA) API to set up and manage a memory pool.
          - [memory_pool_resource_usage_statistics](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool_resource_usage_statistics): Shows how to query resource usage statistics for a memory pool.
          - [memory_pool_threshold](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool_threshold): Shows how to use the stream ordered memory allocation (SOMA) API to set up and manage a memory pool, while defining a threshold to specify an amount of memory to reserve.
          - [memory_pool_trim](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool_trim): Shows how to trim a memory pool to a new size.
          - [ordinary_memory_allocation](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/ordinary_memory_allocation): Shows an ordinary memory allocation.
          - [stream_ordered_memory_allocation](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/stream_ordered_memory_allocation): Shows how to use stream ordered memory allocations.
        - [Unified-Memory-Management](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management) contains the examples from the [Unified memory management](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html) page.
          - [data_prefetching](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/data_prefetching): Shows how to prefetch data in the unified memory space before it is actually needed.
          - [dynamic_unified_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/dynamic_unified_memory): Shows how to dynamically allocate unified memory and use it from both the host and the device.
          - [explicit_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/explicit_memory): Shows how to perform explicit memory management by allocating memory on the device and transferring bytes between the host and the device.
          - [memory_range_attributes](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/memory_range_attributes): Shows how to query attributes of a given memory range.
          - [standard_unified_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/standard_unified_memory): Shows demonstrates how to dynamically allocate unified memory with standard C++ facilities and use it from both the host and the device.
          - [static_unified_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/static_unified_memory): Shows how to statically allocate unified memory and use it from both the host and the device.
          - [unified_memory_advice](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/unified_memory_advice): Shows how to set unified memory runtime hints.
        - [Virtual-Memory-Management](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Virtual-Memory-Management) contains the examples from the [Virtual memory management](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html) page.
          - [virtual_memory](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Virtual-Memory-Management/virtual_memory): Shows how to use HIP's virtual memory management API.
      - [Multi-Device-Management](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management) contains the examples from the [Multi-device management](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html) page.
        - [device_enumeration](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/device_enumeration): Shows how to query the number of devices in the system and how to access them.
        - [device_selection](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/device_selection): Shows how to switch between the different devices in the system and assign work to them.
        - [multi_device_sychronization](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/multi_device_sychronization): Shows how to synchronize multiple devices using HIP events and streams.
        - [p2p_memory_access](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/p2p_memory_access): Shows how to copy data between devices by adding peer-to-peer accesses to the device selection example.
        - [p2p_memory_access_host_staging](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/p2p_memory_access_host_staging): Shows how to copy data between devices by adding peer-to-peer accesses to the device selection example, but explicitly does not enable peer-to-peer access for the devices.
  - [Reference](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference) hosts the examples from the HIP documentation's Reference section.
    - [CUDA-to-HIP-API-Function-Comparison](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference/CUDA-to-HIP-API-Function-Comparison) contains the examples from the [CUDA to HIP API Function Comparison](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/api_syntax.html) page.
      - [block_reduction](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference/CUDA-to-HIP-API-Function-Comparison/block_reduction): Shows a block-reduction kernel written in CUDA.
    - [HIP-Complex-Math-API](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference/HIP-Complex-Math-API) contains the examples from the [HIP complex math API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/complex_math_api.html) page.
      - [complex_math](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference/HIP-Complex-Math-API/complex_math): Shows how to use HIP's complex math API to compute the DFT.
    - [HIP-Math-API](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference/HIP-Math-API) contains the examples from the [HIP math API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html) page.
      - [math](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference/HIP-Math-API/math): Shows how to use HIP's math API to compute the ULP difference.
    - [Low-Precision-Floating-Point-Types](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference/Low-Precision-Floating-Point-Types) contains the examples from the [Low precision floating point types](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html) page.
      - [low_precision_float_fp8](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference/Low-Precision-Floating-Point-Types/low_precision_float_fp8): Shows how to convert a single-precision `float` value to an 8-bit floating-point type and back.
      - [low_precision_float_fp16](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Reference/Low-Precision-Floating-Point-Types/low_precision_float_fp16): Shows how to perform an addition of two 16-bit `__half` values and store the result as single-precision `float`.
  - [Tutorial](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Tutorials) hosts the examples from the HIP documentation's Tutorials section.
    - [graph_api](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Doc/Tutorials/graph_api): Shows how to convert an existing stream-based application to a graph-based application.
- [Dockerfiles](https://github.com/ROCm/rocm-examples/tree/amd-staging/Dockerfiles/) hosts Dockerfiles with ready-to-use environments for the various samples. See [Dockerfiles/README.md](https://github.com/ROCm/rocm-examples/tree/amd-staging/Dockerfiles/README.md) for details.
- [Docs](https://github.com/ROCm/rocm-examples/tree/amd-staging/Docs/)
  - [CONTRIBUTING.md](https://github.com/ROCm/rocm-examples/tree/amd-staging/Docs/CONTRIBUTING.md) contains information on how to contribute to the examples.
- [Libraries](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/)
  - [hipBLAS](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLAS/)
    - [gemm_strided_batched](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLAS/gemm_strided_batched/): Showcases the general matrix product operation with strided and batched matrices.
    - [her](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLAS/her/): Showcases a rank-2 update of a Hermitian matrix with complex values.
    - [scal](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLAS/scal/): Simple program that showcases vector scaling (SCAL) operation.
  - [hipBLASLt](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/)
    - [ext_op_amax](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/ext_op_amax/): Extension API operation for computing absolute maximum values for quantization analysis.
    - [ext_op_layernorm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/ext_op_layernorm/): Extension API layer normalization operation for transformer model optimization.
    - [gemm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm/): Basic general matrix multiplication using core hipBLASLt API.
    - [gemm_alphavec_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_alphavec_ext/): Extension API matrix multiplication with vector alpha scaling for advanced parameter control.
    - [gemm_amax](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_amax/): Matrix multiplication with integrated absolute maximum computation for quantization-aware training.
    - [gemm_amax_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_amax_ext/): Extension API matrix multiplication with integrated AMAX computation using simplified interface.
    - [gemm_amax_with_scale](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_amax_with_scale/): Matrix multiplication combining AMAX computation with input scaling for quantized neural networks.
    - [gemm_amax_with_scale_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_amax_with_scale_ext/): Extension API combining AMAX computation and scaling with high-level interface.
    - [gemm_attr_tciA_tciB](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_attr_tciA_tciB/): Matrix multiplication with tensor core instruction attributes for optimized GPU acceleration.
    - [gemm_batched](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_batched/): Batched matrix multiplication for processing multiple independent matrix operations.
    - [gemm_batched_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_batched_ext/): Extension API batched matrix multiplication with simplified batch management.
    - [gemm_bgradb](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_bgradb/): Matrix multiplication with backward gradient computation for neural network training.
    - [gemm_bias](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_bias/): Matrix multiplication with fused bias addition epilogue for neural network layers.
    - [gemm_bias_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_bias_ext/): Extension API matrix multiplication with fused bias addition using simplified interface.
    - [gemm_bias_swizzle_a_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_bias_swizzle_a_ext/): Extension API with matrix A swizzling and fused bias epilogue for neural network layers.
    - [gemm_clamp_bias](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_clamp_bias/): Matrix multiplication with fused bias addition and clamping activation function.
    - [gemm_dgelu_bgradb](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_dgelu_bgradb/): Matrix multiplication with GELU derivative and backward gradient for transformer training.
    - [gemm_dgelu_bgradb_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_dgelu_bgradb_ext/): Extension API combining GELU derivative and backward gradient computation.
    - [gemm_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_ext/): General matrix multiplication using the hipBLASLt extension API with simplified high-level interface.
    - [gemm_ext_bgradb](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_ext_bgradb/): Extension API matrix multiplication with backward gradient computation for training workflows.
    - [gemm_gelu_aux_bias](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_gelu_aux_bias/): Matrix multiplication with GELU activation, auxiliary output, and bias for transformer models.
    - [gemm_gelu_aux_bias_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_gelu_aux_bias_ext/): Extension API combining GELU activation, auxiliary outputs, and bias addition.
    - [gemm_get_algo_by_index_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_get_algo_by_index_ext/): Matrix multiplication with algorithm selection by numerical index for systematic algorithm exploration.
    - [gemm_get_all_algos](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_get_all_algos/): Comprehensive algorithm enumeration and testing for optimal performance selection.
    - [gemm_get_all_algos_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_get_all_algos_ext/): Extension API comprehensive algorithm testing with simplified interface.
    - [gemm_is_tuned_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_is_tuned_ext/): Checks matrix multiplication tuning status for performance optimization validation.
    - [gemm_mix_precision](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_mix_precision/): Mixed-precision matrix multiplication for balanced performance and accuracy.
    - [gemm_mix_precision_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_mix_precision_ext/): Extension API mixed-precision computation with simplified precision management.
    - [gemm_mix_precision_with_amax_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_mix_precision_with_amax_ext/): Mixed-precision matrix multiplication with AMAX computation for dynamic quantization.
    - [gemm_swish_bias](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_swish_bias/): Matrix multiplication with fused Swish activation function and bias addition.
    - [gemm_swizzle_a](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_swizzle_a/): Matrix multiplication with matrix A swizzling optimization for improved cache efficiency and memory bandwidth.
    - [gemm_swizzle_a_scale_a_b_vector](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_swizzle_a_scale_a_b_vector/): Combined matrix A swizzling and vector scaling optimization for quantized neural networks.
    - [gemm_tuning_splitk_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_tuning_splitk_ext/): GEMM with Split-K tuning optimization for improved parallelization and large matrix performance.
    - [gemm_tuning_wgm_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_tuning_wgm_ext/): GEMM with Workgroup Mapping tuning for optimized GPU resource utilization and load balancing.
    - [gemm_with_scale_a_b](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_with_scale_a_b/): Matrix multiplication with scalar input matrix scaling for quantized computations.
    - [gemm_with_scale_a_b_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_with_scale_a_b_ext/): Extension API matrix multiplication with scalar input matrix scaling using simplified interface.
    - [gemm_with_scale_a_b_vector](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_with_scale_a_b_vector/): Matrix multiplication with vector-based input scaling for per-channel quantization.
    - [gemm_with_TF32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/gemm_with_TF32/): Matrix multiplication using TensorFloat-32 precision for enhanced AI/ML performance acceleration.
    - [groupedgemm_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/groupedgemm_ext/): Grouped matrix multiplication for batching operations with different matrix dimensions per group.
    - [groupedgemm_fixed_mk_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/groupedgemm_fixed_mk_ext/): Grouped GEMM with fixed M,K dimensions and variable N for transformer model optimization.
    - [groupedgemm_get_all_algos_ext](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/groupedgemm_get_all_algos_ext/): Grouped GEMM with comprehensive algorithm testing and multiple grouped instances.
    - [weight_swizzle_padding](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipBLASLt/weight_swizzle_padding/): Tensor manipulation utilities demonstrating weight matrix swizzling with automatic padding for GPU optimization.
  - [hipCUB](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipCUB/)
    - [device_radix_sort](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipCUB/device_radix_sort/): Simple program that showcases `hipcub::DeviceRadixSort::SortPairs`.
    - [device_sum](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipCUB/device_sum/): Simple program that showcases `hipcub::DeviceReduce::Sum`.
  - [hipRAND](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipRAND/)
    - [c_cpp_api](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipRAND/c_cpp_api) Showcases the use of the hipRAND cpp API.
      - [simple_distributions_cpp](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipRAND/c_cpp_api/simple_distributions_cpp) Shows an example for a simple distribution.
    - [device_api](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipRAND/device_api) Showcases the use of the hipRAND device API.
      - [pseudorandom_generations](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipRAND/device_api/pseudorandom_generations) Shows an example for a pseudorandom generator inside a kernel.
      - [quasirandom_generations](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipRAND/device_api/quasirandom_generations) Shows an example for a quasirandom generator inside a kernel.
  - [hipSOLVER](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/)
    - [gels](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/gels/): Solve a linear system of the form $A\times X=B$.
    - [geqrf](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/geqrf/): Program that showcases how to obtain a QR decomposition with the hipSOLVER API.
    - [gesvd](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/gesvd/): Program that showcases how to obtain a singular value decomposition with the hipSOLVER API.
    - [getrf](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/getrf): Program that showcases how to perform a LU factorization with hipSOLVER.
    - [potrf](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/potrf/): Perform Cholesky factorization and solve linear system with result.
    - [syevd](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/syevd/): Program that showcases how to calculate the eigenvalues of a matrix using a divide-and-conquer algorithm in hipSOLVER.
    - [syevdx](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/syevdx/): Shows how to compute a subset of the eigenvalues and the corresponding eigenvectors of a real symmetric matrix A using the Compatibility API of hipSOLVER.
    - [sygvd](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/sygvd/): Showcases how to obtain a solution $(X, \Lambda)$ for a generalized symmetric-definite eigenvalue problem of the form $A \cdot X = B\cdot X \cdot \Lambda$.
    - [syevj](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/syevj): Calculates the eigenvalues and eigenvectors from a real symmetric matrix using the Jacobi method.
    - [syevj_batched](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/syevj_batched): Showcases how to compute the eigenvalues and eigenvectors (via Jacobi method) of each matrix in a batch of real symmetric matrices.
    - [sygvj](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSOLVER/sygvj): Calculates the generalized eigenvalues and eigenvectors from a pair of real symmetric matrices using the Jacobi method.
  - [hipSPARSE](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSPARSE/)
    - [axpyi](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSPARSE/axpyi/): Showcases how to scale a sparse vector and add it to a dense vector.
    - [csrmv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSPARSE/csrmv/): Showcases CSR matrix-vector multiplication with performance optimization.
    - [handle](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSPARSE/handle/): Showcases hipSPARSE library initialization, handle management, and version querying.
    - [hybmv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSPARSE/hybmv/): Showcases HYB (Hybrid) format matrix-vector multiplication for optimized performance.
  - [hipSPARSELt](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSPARSELt/)
    - [spmm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSPARSELt/spmm): Perform a *sparse matrix - dense matrix multiplication*.
    - [spmm_advanced](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipSPARSELt/spmm_advanced): Perform a *sparse matrix - dense matrix multiplication* with a scaling vector, bias addition and an activation function.
  - [hipTensor](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/)
    - [contraction](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction): Tensor contraction operations that compute products over shared dimensions
      - [bilinear](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/bilinear): Bilinear contractions with accumulation support for combining tensor products with existing output data
        - [bf16_f32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/bilinear/bf16_f32): Demonstrates bilinear tensor contraction using BFloat16 data with FP32 computation.
        - [cf32_cf32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/bilinear/cf32_cf32): Shows bilinear tensor contraction with single-precision complex floating-point data.
        - [f16_f32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/bilinear/f16_f32): Demonstrates mixed precision bilinear contraction with FP16 data and FP32 computation.
        - [f32_bf16](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/bilinear/f32_bf16): Shows inverse mixed precision with FP32 data and BFloat16 computation.
        - [f32_f16](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/bilinear/f32_f16): Demonstrates FP32 data storage with FP16 computational precision.
        - [f32_f32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/bilinear/f32_f32): Standard single-precision bilinear tensor contraction implementation.
        - [f64_f32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/bilinear/f64_f32): Shows mixed precision with FP64 data and FP32 computation.
        - [f64_f64](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/bilinear/f64_f64): Demonstrates maximum precision bilinear contraction using double-precision arithmetic.
      - [scale](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/scale): Scale contractions that compute tensor products from scratch without accumulation for cleaner numerical behavior
        - [bf16_f32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/scale/bf16_f32): Shows scale tensor contraction using BFloat16 data with FP32 computation.
        - [cf32_cf32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/scale/cf32_cf32): Demonstrates scale contraction with single-precision complex floating-point data.
        - [f16_f32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/scale/f16_f32): Shows mixed precision scale contraction with FP16 data and FP32 computation.
        - [f32_bf16](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/scale/f32_bf16): Demonstrates inverse mixed precision scale contraction with FP32 data and BFloat16 computation.
        - [f32_f16](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/scale/f32_f16): Shows FP32 data storage with FP16 computational precision for scale contractions.
        - [f32_f32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/scale/f32_f32): Standard single-precision scale tensor contraction implementation.
        - [f64_f32](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/scale/f64_f32): Demonstrates mixed precision scale contraction with FP64 data and FP32 computation.
        - [f64_f64](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/contraction/scale/f64_f64): Shows maximum precision scale contraction using double-precision arithmetic.
    - [elementwise](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/elementwise): Element-by-element tensor operations including arithmetic, permutation, and multi-tensor combinations
      - [binary](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/elementwise/binary): Program that demonstrates elementwise binary operations with tensor permutation.
      - [permute](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/elementwise/permute): Shows how to perform tensor dimension reordering operations.
      - [trinary](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/elementwise/trinary): Shows how to combine three tensors using nested binary operations with permutation.
    - [reduction](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipTensor/reduction): Program that showcases tensor reduction operations using hipTensor.
  - [MIGraphX](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIGraphX/)
    - [migraphx](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIGraphX/migraphx/): Core MIGraphX functionality examples including model parsing, custom operators, and dynamic batch processing.
      - [cpp_dynamic_batch](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIGraphX/migraphx/cpp_dynamic_batch/): Demonstrates running graph programs with dynamic batch sizes using the MIGraphX C++ API.
      - [cpp_parse_load_save](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIGraphX/migraphx/cpp_parse_load_save/): Shows how to parse ONNX models, serialize programs to MessagePack or JSON format, and load saved programs.
      - [custom_op_hip_kernel](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIGraphX/migraphx/custom_op_hip_kernel/): Demonstrates implementing custom operators with HIP kernels for element-wise operations.
      - [custom_op_miopen_kernel](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIGraphX/migraphx/custom_op_miopen_kernel/): Shows how to integrate MIOpen's optimized deep learning primitives as custom operators.
      - [custom_op_rocblas_kernel](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIGraphX/migraphx/custom_op_rocblas_kernel/): Demonstrates integrating rocBLAS linear algebra routines as custom operators.
    - [vision](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIGraphX/vision/): Computer vision inference examples.
      - [cpp_mnist](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIGraphX/vision/cpp_mnist/): Demonstrates MNIST handwritten digit inference with optional FP16/INT8 quantization and multi-target support.
  - [MIVisionX](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIVisionX/)
    - [canny](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIVisionX/canny/): Demonstrates Canny edge detection using OpenVX framework with color space conversion, channel extraction, and configurable hysteresis thresholding.
    - [mv_objdetect](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIVisionX/mv_objdetect/): Showcases real-time object detection using MIVisionX deployment framework with pre-trained YoloV2 Tiny model, including automated model compilation and video decoding integration.
    - [opencv_orb](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/MIVisionX/opencv_orb/): Demonstrates ORB (Oriented FAST and Rotated BRIEF) feature detection using OpenVX with OpenCV extensions for keypoint detection and descriptor computation.
  - [RCCL](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/)
    - [allgather](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/allgather/): Showcases how to collect data from all ranks and distribute the concatenated result to every rank.
    - [allreduce](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/allreduce/): Showcases how to reduce data from all ranks and distribute the result to every rank.
    - [broadcast](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/broadcast/): Showcases how to distribute data from a root rank to all ranks in the communicator.
    - [buffer_registration](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/buffer_registration/): Showcases buffer registration optimization for repeated collective operations to eliminate per-iteration memory management overhead.
    - [device_api](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/device_api/): Showcases RCCL device-side API concepts and the benefits of fusing computation with collective communication operations.
    - [gradient_allreduce](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/gradient_allreduce/): Showcases AllReduce operations in a distributed deep learning training scenario with multiple gradient layers.
    - [reduce](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/reduce/): Showcases how to reduce data from all ranks to a single specified root rank.
    - [reducescatter](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/reducescatter/): Showcases how to reduce data from all ranks and scatter the result chunks to all ranks.
    - [send_recv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RCCL/send_recv/): Showcases point-to-point communication using Send and Recv operations with ring topology.
  - [rocALUTION](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/)
    - [amg](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/amg/): Showcases Smoothed Aggregation Algebraic Multigrid method for solving linear systems with automatic hierarchy construction.
    - [as_precond](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/as_precond/): Showcases Additive Schwarz preconditioner with domain decomposition and two-level preconditioning.
    - [async](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/async/): Showcases asynchronous memory transfers and computations to overlap data movement with computation.
    - [benchmark](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/benchmark/): Showcases performance benchmarking of fundamental linear algebra operations across different matrix formats.
    - [bicgstab](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/bicgstab/): Showcases Bi-Conjugate Gradient Stabilized method for solving nonsymmetric linear systems.
    - [block_precond](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/block_precond/): Showcases block preconditioning techniques for structured linear systems.
    - [cg](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/cg/): Showcases Conjugate Gradient method for solving symmetric positive definite linear systems.
    - [cg_amg](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/cg_amg/): Showcases Conjugate Gradient solver combined with Algebraic Multigrid preconditioning.
    - [cg_rsamg](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/cg_rsamg/): Showcases Conjugate Gradient solver with Rough Smoothed Aggregation Multigrid preconditioning.
    - [cg_saamg](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/cg_saamg/): Showcases Conjugate Gradient solver with Smoothed Aggregation Algebraic Multigrid preconditioning.
    - [cmk](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/cmk/): Showcases Chebyshev-Marker-Krylov methods for eigenvalue problems.
    - [complex](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/complex/): Showcases complex-valued linear solver using Induced Dimension Reduction method.
    - [direct](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/direct/): Showcases direct solver methods for linear systems using factorization techniques.
    - [fcg](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/fcg/): Showcases Flexible Conjugate Gradient method for varying preconditioner scenarios.
    - [fgmres](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/fgmres/): Showcases Flexible Generalized Minimal Residual method with varying preconditioners.
    - [fixed_point](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/fixed_point/): Showcases fixed-point iteration methods for solving linear systems.
    - [gmres](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/gmres/): Showcases Generalized Minimal Residual method for solving nonsymmetric linear systems.
    - [idr](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/idr/): Showcases Induced Dimension Reduction method for solving nonsymmetric linear systems.
    - [itsolve](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/itsolve/): Showcases iterative solver framework and configuration options.
    - [key](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/key/): Showcases key linear algebra operations and solver utilities.
    - [me_preconditioner](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/me_preconditioner/): Showcases Multi-Element preconditioning techniques for enhanced convergence.
    - [mixed_precision](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/mixed_precision/): Showcases mixed-precision defect correction method using different precision levels for inner/outer solvers.
    - [power_method](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/power_method/): Showcases power iteration method for finding dominant eigenvalues and eigenvectors.
    - [sa_amg](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/sa_amg/): Showcases Smoothed Aggregation Algebraic Multigrid method with advanced configuration options.
    - [simple_spmv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/simple_spmv/): Showcases fundamental sparse matrix-vector multiplication operations with format conversion.
    - [sp_precond](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/sp_precond/): Showcases sparse preconditioning techniques for iterative solvers.
    - [stencil](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/stencil/): Showcases stencil-based operations and structured grid computations.
    - [tns](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/tns/): Showcases tensor network solver methods for high-dimensional problems.
    - [ua_amg](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/ua_amg/): Showcases Unsmoothed Aggregation Algebraic Multigrid method for coarse grid construction.
    - [var_precond](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocALUTION/var_precond/): Showcases variable preconditioning techniques for adaptive solver performance.
  - [rocBLAS](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/)
    - [level_1](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_1/): Operations between vectors and vectors.
      - [axpy](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_1/axpy/): Simple program that showcases the AXPY operation.
      - [dot](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_1/dot/): Simple program that showcases dot product.
      - [nrm2](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_1/nrm2/): Simple program that showcases Euclidean norm of a vector.
      - [scal](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_1/scal/): Simple program that showcases vector scaling (SCAL) operation.
      - [swap](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_1/swap/): Showcases exchanging elements between two vectors.
    - [level_2](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_2/): Operations between vectors and matrices.
      - [her](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_2/her/): Showcases a rank-1 update of a Hermitian matrix with complex values.
      - [gemv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_2/gemv/): Showcases the general matrix-vector product operation.
    - [level_3](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_3/): Operations between matrices and matrices.
      - [gemm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_3/gemm/): Showcases the general matrix product operation.
      - [gemm_strided_batched](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocBLAS/level_3/gemm_strided_batched/): Showcases the general matrix product operation with strided and batched matrices.
  - [rocJPEG](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocJPEG/)
    - [rocjpeg_decode](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocJPEG/rocjpeg_decode): Program that showcases decoding of JPEG images.
    - [rocjpeg_decode_batched](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocJPEG/rocjpeg_decode_batched): Program that showcases decoding a batch of JPEG images.
    - [rocjpeg_decode_perf](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocJPEG/rocjpeg_decode_perf): Program that showcases performant decoding of JPEG images.
  - [rocFFT](/Libraries/rocFFT/)
    - [callback](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocFFT/callback/): Program that showcases the use of rocFFT `callback` functionality.
    - [complex_complex](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocFFT/complex_complex/): Program that showcases a Fast Fourier Transform from complex to complex numbers.
    - [complex_real](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocFFT/complex_real/): Program that showcases a Fast Fourier Transform from complex to real numbers.
    - [multi_gpu](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocFFT/multi_gpu/): Program that showcases the use of rocFFT multi-GPU functionality.
    - [real_complex](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocFFT/real_complex/): Program that showcases a Fast Fourier Transform from real to complex numbers.
  - [rocPRIM](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocPRIM/)
    - [block_sum](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocPRIM/block_sum/): Simple program that showcases `rocprim::block_reduce` with an addition operator.
    - [device_sum](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocPRIM/device_sum/): Simple program that showcases `rocprim::reduce` with an addition operator.
  - [rocProfiler-SDK](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/)
    - [api_buffered_tracing](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/api_buffered_tracing/): Demonstrates buffered tracing of HIP/HSA API calls, kernel dispatches, memory copies, and scratch memory usage with batch processing via callbacks.
    - [api_callback_tracing](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/api_callback_tracing/): Showcases direct callback-based tracing of HIP, HSA, and ROCTX API calls with synchronous event processing and dynamic pause/resume control.
    - [code_object_isa_decode](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/code_object_isa_decode/): Demonstrates intercepting GPU kernel binaries, decoding their instruction set architecture (ISA), and performing instruction-level analysis.
    - [code_object_tracing](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/code_object_tracing/): Showcases monitoring the lifecycle of GPU code objects, including load/unload events and kernel symbol registration.
    - [counter_collection](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/counter_collection/): Hardware counter collection from kernel dispatches and devices.
      - [buffer](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/counter_collection/buffer/): Showcases buffer-based counter collection with batch processing via callbacks.
      - [buffer_device_serialization](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/counter_collection/buffer_device_serialization/): Demonstrates buffer-based counter collection across multiple devices with per-device serialization.
      - [callback](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/counter_collection/callback/): Showcases direct callback-based counter collection with immediate, synchronous processing.
      - [device_profiling](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/counter_collection/device_profiling/): Demonstrates asynchronous, device-level counter sampling on a separate thread.
      - [device_profiling_sync](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/counter_collection/device_profiling_sync/): Showcases synchronous, on-demand device-level counter collection with a high-level sampler class.
      - [print_functional_counters](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/counter_collection/print_functional_counters/): Functional test that systematically validates all available hardware counters on a GPU agent.
    - [external_correlation_id_request](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/external_correlation_id_request/): Demonstrates associating custom correlation IDs with asynchronous GPU operations to link them back to CPU context.
    - [intercept_table](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/intercept_table/): Showcases direct control over the HIP runtime API dispatch table by replacing function pointers with custom wrappers.
    - [openmp_target](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/openmp_target/): Demonstrates tracing OpenMP target offloading applications with OMPT events, ROCTX markers, and GPU activities.
    - [pc_sampling](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/pc_sampling/): Showcases PC (Program Counter) sampling to statistically profile where time is spent within GPU kernels for performance analysis.
    - [thread_trace](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocProfiler-SDK/thread_trace/): Demonstrates instruction-by-instruction wavefront execution tracing with PC and latency information for detailed performance hotspot analysis.
  - [hipFFT](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipFFT/)
    - [multi_gpu](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipFFT/multi_gpu/): Program that showcases the use of hipFFT multi-GPU functionality.
    - [plan_d2z](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipFFT/plan_d2z): Forward fast Fourier transform for 1D, 2D, and 3D real input using a simple plan in hipFFT.
    - [plan_z2z](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/hipFFT/plan_z2z): Forward fast Fourier transform for 1D, 2D, and 3D complex input using a simple plan in hipFFT.
  - [rocRAND](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocRAND/)
    - [c_cpp_api](/Libraries/rocRAND/c_cpp_api/): rocRAND's C/C++ API examples.
      - [simple_distributions_cpp](/Libraries/rocRAND/c_cpp_api/simple_distributions_cpp/): A command-line app to compare random number generation on the CPU and on the GPU with rocRAND.
    - [device_api](/Libraries/rocRAND/device_api/): rocRAND's device API examples.
      - [pseudorandom_generations](/Libraries/rocRAND/device_api/pseudorandom_generations): Simple program that shows how to generate random values with rocRAND's pseudorandom generators.
  - [rocSOLVER](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSOLVER/)
    - [getf2](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSOLVER/getf2): Program that showcases how to perform a LU factorization with rocSOLVER.
    - [getri](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSOLVER/getri): Program that showcases matrix inversion by LU-decomposition using rocSOLVER.
    - [syev](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSOLVER/syev): Shows how to compute the eigenvalues and eigenvectors from a symmetrical real matrix.
    - [syev_batched](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSOLVER/syev_batched): Shows how to compute the eigenvalues and eigenvectors for each matrix in a batch of real symmetric matrices.
    - [syev_strided_batched](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSOLVER/syev_strided_batched): Shows how to compute the eigenvalues and eigenvectors for multiple symmetrical real matrices, that are stored with an arbitrary stride.
  - [rocSPARSE](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/)
    - [level_1](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_1/): Operations between sparse vectors and dense vectors.
      - [axpyi](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_1/axpyi/): Showcases how to scale a sparse vector and add it to a dense vector.
      - [doti](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_1/doti/): Showcases a dot product of a sparse vector with a dense vector.
      - [gthr](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_1/gthr/): Showcases how to gather elements from a dense vector and store them into a sparse vector.
      - [roti](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_1/roti/): Showcases a Givens rotation to a dense and a sparse vector.
      - [sctr](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_1/sctr/): Showcases how to scatter elements in a sparse vector into a dense vector.
    - [level_2](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/): Operations between sparse matrices and dense vectors.
      - [bsrmv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/bsrmv/): Showcases a sparse matrix-vector multiplication using BSR storage format.
      - [bsrxmv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/bsrxmv/): Showcases a masked sparse matrix-vector multiplication using BSR storage format.
      - [bsrsv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/bsrsv/): Showcases how to solve a linear system of equations whose coefficients are stored in a BSR sparse triangular matrix.
      - [coomv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/coomv/): Showcases a sparse matrix-vector multiplication using COO storage format.
      - [csritsv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/csritsv/): Showcases how to find an iterative solution with the Jacobi method for a linear system of equations whose coefficients are stored in a CSR sparse triangular matrix.
      - [csrmv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/csrmv/): Showcases a sparse matrix-vector multiplication using CSR storage format.
      - [csrsv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/csrsv/): Showcases how to solve a linear system of equations whose coefficients are stored in a CSR sparse triangular matrix.
      - [ellmv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/ellmv/): Showcases a sparse matrix-vector multiplication using ELL storage format.
      - [gebsrmv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/gebsrmv/): Showcases a sparse matrix-dense vector multiplication using GEBSR storage format.
      - [gemvi](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/gemvi/): Showcases a dense matrix-sparse vector multiplication.
      - [spitsv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/spitsv/): Showcases how to solve iteratively a linear system of equations whose coefficients are stored in a CSR sparse triangular matrix.
      - [spmv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/spmv/): Showcases a general sparse matrix-dense vector multiplication.
      - [spsv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_2/spsv/): Showcases how to solve a linear system of equations whose coefficients are stored in a sparse triangular matrix.
    - [level_3](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/): Operations between sparse and dense matrices.
      - [bsrmm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/bsrmm/): Showcases a sparse matrix-matrix multiplication using BSR storage format.
      - [bsrsm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/bsrsm): Showcases how to solve a linear system of equations whose coefficients are stored in a BSR sparse triangular matrix, with solution and right-hand side stored in dense matrices.
      - [csrmm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/csrmm/): Showcases a sparse matrix-matrix multiplication using CSR storage format.
      - [csrsm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/csrsm): Showcases how to solve a linear system of equations whose coefficients are stored in a CSR sparse triangular matrix, with solution and right-hand side stored in dense matrices.
      - [gebsrmm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/gebsrmm/): Showcases a sparse matrix-matrix multiplication using GEBSR storage format.
      - [gemmi](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/gemmi/): Showcases a dense matrix sparse matrix multiplication using CSR storage format.
      - [sddmm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/sddmm/): Showcases a sampled dense-dense matrix multiplication using CSR storage format.
      - [spmm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/spmm/): Showcases a sparse matrix-dense matrix multiplication.
      - [spsm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/level_3/spsm/): Showcases a sparse triangular linear system solver using CSR storage format.
    - [preconditioner](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/preconditioner/): Manipulations on sparse matrices to obtain sparse preconditioner matrices.
      - [bsric0](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/preconditioner/bsric0/): Shows how to compute the incomplete Cholesky decomposition of a Hermitian positive-definite sparse BSR matrix.
      - [bsrilu0](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/preconditioner/bsrilu0/): Showcases how to obtain the incomplete LU decomposition of a sparse BSR square matrix.
      - [csric0](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/preconditioner/csric0/): Shows how to compute the incomplete Cholesky decomposition of a Hermitian positive-definite sparse CSR matrix.
      - [csrilu0](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/preconditioner/csrilu0/): Showcases how to obtain the incomplete LU decomposition of a sparse CSR square matrix.
      - [csritilu0](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/preconditioner/csritilu0/): Showcases how to obtain iteratively the incomplete LU decomposition of a sparse CSR square matrix.
      - [gpsv](https://github.com/amd/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/preconditioner/gpsv/): Shows how to compute the solution of pentadiagonal linear system.
      - [gtsv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocSPARSE/preconditioner/gtsv/): Shows how to compute the solution of a tridiagonal linear system.
  - [rocThrust](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocThrust/)
    - [device_ptr](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocThrust/device_ptr/): Simple program that showcases the usage of the `thrust::device_ptr` template.
    - [norm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocThrust/norm/): An example that computes the Euclidean norm of a `thrust::device_vector`.
    - [reduce_sum](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocThrust/reduce_sum/): An example that computes the sum of a `thrust::device_vector` integer vector using the `thrust::reduce()` generalized summation and the `thrust::plus` operator.
    - [remove_points](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocThrust/remove_points/): Simple program that demonstrates the usage of the `thrust` random number generation, host vector, generation, tuple, zip iterator, and conditional removal templates. It generates a number of random points in a unit square and then removes all of them outside the unit circle.
    - [saxpy](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocThrust/saxpy/): Simple program that implements the SAXPY operation (`y[i] = a * x[i] + y[i]`) using rocThrust and showcases the usage of the vector and functor templates and of `thrust::fill` and `thrust::transform` operations.
    - [vectors](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocThrust/vectors/): Simple program that showcases the `host_vector` and the `device_vector` of rocThrust.
  - [rocWMMA](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/)
    - [hiprtc_gemm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/hiprtc_gemm/) : Showcases a simple matrix-matrix multiplication via hipRTC.
    - [perf_dgemm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/perf_dgemm/) : Showcases a performant double-precision matrix-matrix multiplication.
    - [perf_hgemm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/perf_hgemm/) : Showcases a performant half-precision matrix-matrix multiplication.
    - [perf_sgemm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/perf_sgemm/) : Showcases a performant single-precision matrix-matrix multiplication.
    - [simple_dgemm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/simple_dgemm/) : Showcases a simple double-precision matrix-matrix multiplication.
    - [simple_dgemv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/simple_dgemv/) : Showcases a simple double-precision matrix-vector multiplication.
    - [simple_dlrm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/simple_dlrm/) : Showcases a simple deep learning recommendation model (DLRM) computation.
    - [simple_hgemm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/simple_hgemm/) : Showcases a simple half-precision matrix-matrix multiplication.
    - [simple_sgemm](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/simple_sgemm/) : Showcases a simple single-precision matrix-matrix multiplication.
    - [simple_sgemv](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/rocWMMA/simple_sgemv/) : Showcases a simple single-precision matrix-vector multiplication.
  - [RPP](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RPP/)
    - [box_filter](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RPP/box_filter/): Showcases how to apply a box filter (mean filter) to images using spatial convolution with configurable kernel sizes.
    - [brightness](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RPP/brightness/): Showcases how to adjust image brightness using linear transformation with alpha and beta parameters.
    - [contrast](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RPP/contrast/): Showcases how to adjust image contrast by scaling pixel values around a center point.
    - [flip](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RPP/flip/): Showcases how to flip images horizontally, vertically, or both to create mirror reflections.
    - [gamma_correction](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RPP/gamma_correction/): Showcases how to apply gamma correction using power-law transformation to adjust image brightness and contrast.
    - [resize](https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RPP/resize/): Showcases how to resize images to specified dimensions using nearest neighbor or bilinear interpolation.
- [Tools](https://github.com/ROCm/rocm-examples/tree/amd-staging/Tools/): Showcases the ROCm tools for debugging and performance analysis.
  - [ROCgdb](https://github.com/ROCm/rocm-examples/tree/amd-staging/Tools/ROCgdb): Shows how to use ROCgdb for GPU debugging.
  - [rocprof-compute](https://github.com/ROCm/rocm-examples/tree/amd-staging/Tools/rocprof-compute): Shows how to use the ROCm Compute Profiler.
  - [rocprof-systems](https://github.com/ROCm/rocm-examples/tree/amd-staging/Tools/rocprof-systems): Demonstrates how to use the ROCm Systems Profiler.
  - [rocprofv3](https://github.com/ROCm/rocm-examples/tree/amd-staging/Tools/rocprofv3): Illustrates how to use the `rocprofv3` profiler.
- [Tutorials](https://github.com/ROCm/rocm-examples/tree/amd-staging/Tutorials/): Showcases HIP Documentation Tutorials.
  - [reduction](https://github.com/ROCm/rocm-examples/tree/amd-staging/Tutorials/reduction/): Showcases a reduction tutorial for HIP Documentation.
