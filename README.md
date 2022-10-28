# ROCm Examples
This project is currently unsupported and in an early testing stage. Feedback on the contents of this repository is appreciated.
## Repository Contents
- [Common](/Common/) contains common utility functionality shared between the examples.
- [HIP-Basic](/HIP-Basic/) hosts self-contained recipes showcasing HIP runtime functionality.
    - [assembly_to_executable](/HIP-Basic/assembly_to_executable): Program and accompanying build systems that show how to manually compile and link a HIP application from host and device code.
    - [bandwidth](/HIP-Basic/bandwidth): Program that measures memory bandwidth from host to device, device to host, and device to device.
    - [bit_extract](/HIP-Basic/bit_extract): Program that showcases how to use HIP built-in bit extract.
    - [device_query](/HIP-Basic/device_query): Program that showcases how properties from the device may be queried.
    - [dynamic_shared](/HIP-Basic/dynamic_shared): Program that showcases how to use dynamic shared memory with the help of a simple matrix transpose kernel.
    - [events](/HIP-Basic/events/): Measuring execution time and synchronizing with HIP events.
    - [hello_world](/HIP-Basic/hello_world): Simple program that showcases launching kernels and printing from the device.
    - [hipify](/HIP-Basic/hipify): Simple program and build definitions that showcase automatically converting a CUDA `.cu` source into portable HIP `.hip` source.
    - [llvm_ir_to_executable](/HIP-Basic/llvm_ir_to_executable): Shows how to create a HIP executable from LLVM IR.
    - [matrix_multiplication](/HIP-Basic/matrix_multiplication/): Multiply two dynamically sized matrices utilizing shared memory.
    - [multi_gpu_data_transfer](/HIP-Basic/multi_gpu_data_transfer/): Performs two matrix transposes on two different devices (one on each) to showcase how to use peer-to-peer communication among devices.
    - [occupancy](/HIP-Basic/occupancy/): Shows how to find optimal configuation parameters for a kernel launch with maximum occupancy.
    - [runtime_compilation](/HIP-Basic/runtime_compilation/): Simple program that showcases how to use HIP runtime compilation (hipRTC) to compile a kernel and launch it on a device.
    - [saxpy](/HIP-Basic/saxpy/): Implements the $Y_i=aX_i+Y_i$ kernel and explains basic HIP functionality.
    - [shared_memory](/HIP-Basic/shared_memory/): Showcases how to use static shared memory by implementing a simple matrix transpose kernel.
    - [streams](/HIP-Basic/streams/): Program that showcases usage of multiple streams each with their own tasks.
    - [warp_shuffle](/HIP-Basic/warp_shuffle/): Uses a simple matrix transpose kernel to showcase how to use warp shuffle operations.
- [Dockerfiles](/Dockerfiles/) hosts Dockerfiles with ready-to-use environments for the various samples. See [Dockerfiles/README.md](/Dockerfiles/README.md) for details.
- [Docs](/Docs/)
    - [CONTRIBUTING.md](/Docs/CONTRIBUTING.md) contains information on how to contribute to the examples.
- [Libraries](/Libraries/)
    - [hipCUB](/Libraries/hipCUB/)
        - [device_radix_sort](/Libraries/hipCUB/device_radix_sort/): Simple program that showcases `hipcub::DeviceRadixSort::SortPairs`.
        - [device_sum](/Libraries/hipCUB/device_sum/): Simple program that showcases `hipcub::DeviceReduce::Sum`.
    - [rocPRIM](/Libraries/rocPRIM/)
        - [block_sum](/Libraries/rocPRIM/block_sum/): Simple program that showcases `rocprim::block_reduce` with an addition operator.
        - [device_sum](/Libraries/rocPRIM/device_sum/): Simple program that showcases `rocprim::reduce` with an addition operator.
    - [rocRAND](/Libraries/rocRAND/)
        - [simple_distributions_cpp](/Libraries/rocRAND/simple_distributions_cpp/): A command-line app to compare random number generation on the CPU and on the GPU with rocRAND.
    - [rocThrust](/Libraries/rocThrust/)
        - [device_ptr](/Libraries/rocThrust/device_ptr/): Simple program that showcases the usage of the `thrust::device_ptr` template.
        - [norm](/Libraries/rocThrust/norm/): An example that computes the Euclidean norm of a `thrust::device_vector`.
        - [reduce_sum](/Libraries/rocThrust/reduce_sum/): An example that computes the sum of a `thrust::device_vector` integer vector using the `thrust::reduce()` generalized summation and the `thrust::plus` operator.
        - [remove_points](/Libraries/rocThrust/remove_points/): Simple program that demonstrates the usage of the `thrust` random number generation, host vector, generation, tuple, zip iterator, and conditional removal templates. It generates a number of random points in a unit square and then removes all of them outside the unit circle.
        - [saxpy](/Libraries/rocThrust/saxpy/): Simple program that implements the SAXPY operation (`Y[i] = a * X[i] + Y[i]`) using rocThrust and showcases the usage of the vector and functor templates and of `thrust::fill` and `thrust::transform` operations.
        - [vectors](/Libraries/rocThrust/vectors/): Simple program that showcases the `host_vector` and the `device_vector` of rocThrust.

## Prerequisites
### Linux
- [CMake](https://cmake.org/download/) (at least version 3.21)
- A number of examples also support building via  GNU Make - available through the distribution's package manager
- [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.1.3/page/Overview_of_ROCm_Installation_Methods.html) (at least version 5.x.x)
- For example-specific prerequisites, see the example subdirectories.

### Windows
- [Visual Studio](https://visualstudio.microsoft.com/) 2019 or 2022 with the "Desktop Development with C++" workload
- ROCm toolchain for Windows (No public release yet)
  - The Visual Studio ROCm extension needs to be installed to build with the solution files.
- [CMake](https://cmake.org/download/) (optional, to build with CMake. Requires at least version 3.21)
- [Ninja](https://ninja-build.org/) (optional, to build with CMake)

## Building the example suite
### Linux
These instructions assume that the prerequisites for every example are installed on the system.

#### CMake
See [CMake build options](#cmake-build-options) for an overview of build options.
- `$ git clone https://github.com/amd/rocm-examples.git`
- `$ cd rocm-examples`
- `$ cmake -S . -B build` (on ROCm) or `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (on CUDA)
- `$ cmake --build build`

#### Make
Beware that only a subset of the examples support building via Make.
- `$ git clone https://github.com/amd/rocm-examples.git`
- `$ cd rocm-examples`
- `$ make` (on ROCm) or `$ make GPU_RUNTIME=CUDA` (on CUDA)

### Linux with Docker
Alternatively, instead of installing the prerequisites on the system, the [Dockerfiles](/Dockerfiles/) in this repository can be used to build images that provide all required prerequisites. Note, that the ROCm kernel GPU driver still needs to be installed on the host system.

The following instructions showcase building the Docker image and full example suite inside the container using CMake:
- `$ git clone https://github.com/amd/rocm-examples.git`
- `$ cd rocm-examples/Dockerfiles`
- `$ docker build . -t rocm-examples -f hip-libraries-rocm-ubuntu.Dockerfile` (on ROCm) or `$ docker build . -t rocm-examples -f hip-libraries-cuda-ubuntu.Dockerfile` (on CUDA)
- `$ docker run -it --device /dev/kfd --device /dev/dri rocm-examples bash` (on ROCm) or `$ docker run -it --gpus=all rocm-examples bash` (on CUDA)
- `# git clone https://github.com/amd/rocm-examples.git`
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
    - The exutables will be created in a subfolder named "Debug" or "Release" inside the project folder.
- The HIP specific project settings like the GPU architectures targeted can be set on the `General [AMD HIP C++]` tab of project properties.

#### CMake
First, clone the repository and go to the source directory.

```shell
git clone https://github.com/amd/rocm-examples.git
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
