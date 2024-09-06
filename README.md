# ROCm Examples

This repository is a collection of examples to enable new users to start using
ROCm, as well as provide more advanced examples for experienced users.

The examples are structured in several categories:

- [HIP-Basic](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/) showcases some basic functionality without any additional dependencies
- [Libraries](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/) contains examples for ROCm-libraries, that provide higher-level functionality
- [Applications](https://github.com/ROCm/rocm-examples/tree/develop/Applications/) showcases some common applications, using HIP to accelerate them

- [AI](https://github.com/ROCm/rocm-examples/tree/develop/AI/) contains instructions on how to use ROCm for AI
- [Tutorials](https://github.com/ROCm/rocm-examples/tree/develop/Tutorials/) contains the code accompanying the HIP Tutorials that can be found in [the HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/saxpy.html).

For a full overview over the examples see the section [repository contents](#repository-contents).

## Prerequisites

### Linux

- [CMake](https://cmake.org/download/) (at least version 3.21)
- A number of examples also support building via  GNU Make - available through the distribution's package manager
- [ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (at least version 6.x.x)
- For example-specific prerequisites, see the example subdirectories.

### Windows

- [Visual Studio](https://visualstudio.microsoft.com/) 2019 or 2022 with the "Desktop Development with C++" workload
- [HIP SDK for Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html)
  - The Visual Studio ROCm extension needs to be installed to build with the solution files.
- [CMake](https://cmake.org/download/) (optional, to build with CMake. Requires at least version 3.21)
- [Ninja](https://ninja-build.org/) (optional, to build with CMake)

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

Alternatively, instead of installing the prerequisites on the system, the [Dockerfiles](https://github.com/ROCm/rocm-examples/tree/develop/Dockerfiles/) in this repository can be used to build images that provide all required prerequisites. Note, that the ROCm kernel GPU driver still needs to be installed on the host system.

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

- [AI](https://github.com/ROCm/rocm-examples/tree/develop/AI/MIGraphX/Quantization) Showcases the functionality for executing quantized models using Torch-MIGraphX.
- [Applications](https://github.com/ROCm/rocm-examples/tree/develop/Applications/) groups a number of examples ... .
  - [bitonic_sort](https://github.com/ROCm/rocm-examples/tree/develop/Applications/bitonic_sort/): Showcases how to order an array of $n$ elements using a GPU implementation of the bitonic sort.
  - [convolution](https://github.com/ROCm/rocm-examples/tree/develop/Applications/convolution/): A simple GPU implementation for the calculation of discrete convolutions.
  - [floyd_warshall](https://github.com/ROCm/rocm-examples/tree/develop/Applications/floyd_warshall/): Showcases a GPU implementation of the Floyd-Warshall algorithm for finding shortest paths in certain types of graphs.
  - [histogram](https://github.com/ROCm/rocm-examples/tree/develop/Applications/histogram/): Histogram over a byte array with memory bank optimization.
  - [monte_carlo_pi](https://github.com/ROCm/rocm-examples/tree/develop/Applications/monte_carlo_pi/): Monte Carlo estimation of $\pi$ using hipRAND for random number generation and hipCUB for evaluation.
  - [prefix_sum](https://github.com/ROCm/rocm-examples/tree/develop/Applications/prefix_sum/): Showcases a GPU implementation of a prefix sum with a 2-kernel scan algorithm.
- [Common](https://github.com/ROCm/rocm-examples/tree/develop/Common/) contains common utility functionality shared between the examples.
- [HIP-Basic](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/) hosts self-contained recipes showcasing HIP runtime functionality.
  - [assembly_to_executable](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/assembly_to_executable): Program and accompanying build systems that show how to manually compile and link a HIP application from host and device code.
  - [bandwidth](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/bandwidth): Program that measures memory bandwidth from host to device, device to host, and device to device.
  - [bit_extract](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/bit_extract): Program that showcases how to use HIP built-in bit extract.
  - [device_globals](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/device_globals): Show cases how to set global variables on the device from the host.
  - [device_query](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/device_query): Program that showcases how properties from the device may be queried.
  - [dynamic_shared](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/dynamic_shared): Program that showcases how to use dynamic shared memory with the help of a simple matrix transpose kernel.
  - [events](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/events/): Measuring execution time and synchronizing with HIP events.
  - [gpu_arch](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/gpu_arch/): Program that showcases how to implement GPU architecture-specific code.
  - [hello_world](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/hello_world): Simple program that showcases launching kernels and printing from the device.
  - [hello_world_cuda](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/hello_world_cuda): Simple HIP program that showcases setting up CMake to target the CUDA platform.
  - [hipify](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/hipify): Simple program and build definitions that showcase automatically converting a CUDA `.cu` source into portable HIP `.hip` source.
  - [llvm_ir_to_executable](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/llvm_ir_to_executable): Shows how to create a HIP executable from LLVM IR.
  - [inline_assembly](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/inline_assembly/): Program that showcases how to use inline assembly in a portable manner.
  - [matrix_multiplication](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/matrix_multiplication/): Multiply two dynamically sized matrices utilizing shared memory.
  - [module_api](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/module_api/): Shows how to load and execute a HIP module in runtime.
  - [moving_average](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/moving_average/): Simple program that demonstrates parallel computation of a moving average of one-dimensional data.
  - [multi_gpu_data_transfer](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/multi_gpu_data_transfer/): Performs two matrix transposes on two different devices (one on each) to showcase how to use peer-to-peer communication among devices.
  - [occupancy](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/occupancy/): Shows how to find optimal configuration parameters for a kernel launch with maximum occupancy.
  - [opengl_interop](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/opengl_interop): Showcases how to share resources and computation between HIP and OpenGL.
  - [runtime_compilation](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/runtime_compilation/): Simple program that showcases how to use HIP runtime compilation (hipRTC) to compile a kernel and launch it on a device.
  - [saxpy](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/saxpy/): Implements the $y_i=ax_i+y_i$ kernel and explains basic HIP functionality.
  - [shared_memory](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/shared_memory/): Showcases how to use static shared memory by implementing a simple matrix transpose kernel.
  - [static_device_library](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/static_device_library): Shows how to create a static library containing device functions, and how to link it with an executable.
  - [static_host_library](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/static_host_library): Shows how to create a static library containing HIP host functions, and how to link it with an executable.
  - [streams](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/streams/): Program that showcases usage of multiple streams each with their own tasks.
  - [texture_management](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/texture_management/): Shows the usage of texture memory.
  - [vulkan_interop](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/vulkan_interop): Showcases how to share resources and computation between HIP and Vulkan.
  - [warp_shuffle](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/warp_shuffle/): Uses a simple matrix transpose kernel to showcase how to use warp shuffle operations.
- [Dockerfiles](https://github.com/ROCm/rocm-examples/tree/develop/Dockerfiles/) hosts Dockerfiles with ready-to-use environments for the various samples. See [Dockerfiles/README.md](https://github.com/ROCm/rocm-examples/tree/develop/Dockerfiles/README.md) for details.
- [Docs](https://github.com/ROCm/rocm-examples/tree/develop/Docs/)
  - [CONTRIBUTING.md](https://github.com/ROCm/rocm-examples/tree/develop/Docs/CONTRIBUTING.md) contains information on how to contribute to the examples.
- [Libraries](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/)
  - [hipBLAS](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipBLAS/)
    - [gemm_strided_batched](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipBLAS/gemm_strided_batched/): Showcases the general matrix product operation with strided and batched matrices.
    - [her](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipBLAS/her/): Showcases a rank-2 update of a Hermitian matrix with complex values.
    - [scal](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipBLAS/scal/): Simple program that showcases vector scaling (SCAL) operation.
  - [hipCUB](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipCUB/)
    - [device_radix_sort](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipCUB/device_radix_sort/): Simple program that showcases `hipcub::DeviceRadixSort::SortPairs`.
    - [device_sum](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipCUB/device_sum/): Simple program that showcases `hipcub::DeviceReduce::Sum`.
  - [hipSOLVER](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/)
    - [gels](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/gels/): Solve a linear system of the form $A\times X=B$.
    - [geqrf](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/geqrf/): Program that showcases how to obtain a QR decomposition with the hipSOLVER API.
    - [gesvd](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/gesvd/): Program that showcases how to obtain a singular value decomposition with the hipSOLVER API.
    - [getrf](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/getrf): Program that showcases how to perform a LU factorization with hipSOLVER.
    - [potrf](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/potrf/): Perform Cholesky factorization and solve linear system with result.
    - [syevd](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/syevd/): Program that showcases how to calculate the eigenvalues of a matrix using a divide-and-conquer algorithm in hipSOLVER.
    - [syevdx](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/syevdx/): Shows how to compute a subset of the eigenvalues and the corresponding eigenvectors of a real symmetric matrix A using the Compatibility API of hipSOLVER.
    - [sygvd](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/sygvd/): Showcases how to obtain a solution $(X, \Lambda)$ for a generalized symmetric-definite eigenvalue problem of the form $A \cdot X = B\cdot X \cdot \Lambda$.
    - [syevj](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/syevj): Calculates the eigenvalues and eigenvectors from a real symmetric matrix using the Jacobi method.
    - [syevj_batched](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/syevj_batched): Showcases how to compute the eigenvalues and eigenvectors (via Jacobi method) of each matrix in a batch of real symmetric matrices.
    - [sygvj](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipSOLVER/sygvj): Calculates the generalized eigenvalues and eigenvectors from a pair of real symmetric matrices using the Jacobi method.
  - [rocBLAS](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/)
    - [level_1](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_1/): Operations between vectors and vectors.
      - [axpy](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_1/axpy/): Simple program that showcases the AXPY operation.
      - [dot](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_1/dot/): Simple program that showcases dot product.
      - [nrm2](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_1/nrm2/): Simple program that showcases Euclidean norm of a vector.
      - [scal](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_1/scal/): Simple program that showcases vector scaling (SCAL) operation.
      - [swap](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_1/swap/): Showcases exchanging elements between two vectors.
    - [level_2](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_2/): Operations between vectors and matrices.
      - [her](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_2/her/): Showcases a rank-1 update of a Hermitian matrix with complex values.
      - [gemv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_2/gemv/): Showcases the general matrix-vector product operation.
    - [level_3](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_3/): Operations between matrices and matrices.
      - [gemm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_3/gemm/): Showcases the general matrix product operation.
      - [gemm_strided_batched](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocBLAS/level_3/gemm_strided_batched/): Showcases the general matrix product operation with strided and batched matrices.
  - [rocFFT](/Libraries/rocFFT/)
    - [callback](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocFFT/callback/): Program that showcases the use of rocFFT `callback` functionality.
    - [complex_complex](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocFFT/complex_complex/): Program that showcases a Fast Fourier Transform from complex to complex numbers.
    - [complex_real](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocFFT/complex_real/): Program that showcases a Fast Fourier Transform from complex to real numbers.
    - [multi_gpu](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocFFT/multi_gpu/): Program that showcases the use of rocFFT multi-GPU functionality.
    - [real_complex](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocFFT/real_complex/): Program that showcases a Fast Fourier Transform from real to complex numbers.
  - [rocPRIM](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocPRIM/)
    - [block_sum](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocPRIM/block_sum/): Simple program that showcases `rocprim::block_reduce` with an addition operator.
    - [device_sum](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocPRIM/device_sum/): Simple program that showcases `rocprim::reduce` with an addition operator.
  - [hipFFT](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipFFT/)
    - [multi_gpu](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipFFT/multi_gpu/): Program that showcases the use of hipFFT multi-GPU functionality.
    - [plan_d2z](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipFFT/plan_d2z): Forward fast Fourier transform for 1D, 2D, and 3D real input using a simple plan in hipFFT.
    - [plan_z2z](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/hipFFT/plan_z2z): Forward fast Fourier transform for 1D, 2D, and 3D complex input using a simple plan in hipFFT.
  - [rocRAND](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocRAND/)
    - [simple_distributions_cpp](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocRAND/simple_distributions_cpp/): A command-line app to compare random number generation on the CPU and on the GPU with rocRAND.
  - [rocSOLVER](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSOLVER/)
    - [getf2](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSOLVER/getf2): Program that showcases how to perform a LU factorization with rocSOLVER.
    - [getri](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSOLVER/getri): Program that showcases matrix inversion by LU-decomposition using rocSOLVER.
    - [syev](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSOLVER/syev): Shows how to compute the eigenvalues and eigenvectors from a symmetrical real matrix.
    - [syev_batched](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSOLVER/syev_batched): Shows how to compute the eigenvalues and eigenvectors for each matrix in a batch of real symmetric matrices.
    - [syev_strided_batched](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSOLVER/syev_strided_batched): Shows how to compute the eigenvalues and eigenvectors for multiple symmetrical real matrices, that are stored with an arbitrary stride.
  - [rocSPARSE](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/)
    - [level_2](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/): Operations between sparse matrices and dense vectors.
      - [bsrmv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/bsrmv/): Showcases a sparse matrix-vector multiplication using BSR storage format.
      - [bsrxmv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/bsrxmv/): Showcases a masked sparse matrix-vector multiplication using BSR storage format.
      - [bsrsv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/bsrsv/): Showcases how to solve a linear system of equations whose coefficients are stored in a BSR sparse triangular matrix.
      - [coomv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/coomv/): Showcases a sparse matrix-vector multiplication using COO storage format.
      - [csritsv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/csritsv/): Showcases how find an iterative solution with the Jacobi method for a linear system of equations whose coefficients are stored in a CSR sparse triangular matrix.
      - [csrmv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/csrmv/): Showcases a sparse matrix-vector multiplication using CSR storage format.
      - [csrsv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/csrsv/): Showcases how to solve a linear system of equations whose coefficients are stored in a CSR sparse triangular matrix.
      - [ellmv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/ellmv/): Showcases a sparse matrix-vector multiplication using ELL storage format.
      - [gebsrmv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/gebsrmv/): Showcases a sparse matrix-dense vector multiplication using GEBSR storage format.
      - [gemvi](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/gemvi/): Showcases a dense matrix-sparse vector multiplication.
      - [spitsv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/spitsv/): Showcases how to solve iteratively a linear system of equations whose coefficients are stored in a CSR sparse triangular matrix.
      - [spmv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/spmv/): Showcases a general sparse matrix-dense vector multiplication.
      - [spsv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/spsv/): Showcases how to solve a linear system of equations whose coefficients are stored in a sparse triangular matrix.
    - [level_3](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/): Operations between sparse and dense matrices.
      - [bsrmm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/bsrmm/): Showcases a sparse matrix-matrix multiplication using BSR storage format.
      - [bsrsm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/bsrsm): Showcases how to solve a linear system of equations whose coefficients are stored in a BSR sparse triangular matrix, with solution and right-hand side stored in dense matrices.
      - [csrmm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/csrmm/): Showcases a sparse matrix-matrix multiplication using CSR storage format.
      - [csrsm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/csrsm): Showcases how to solve a linear system of equations whose coefficients are stored in a CSR sparse triangular matrix, with solution and right-hand side stored in dense matrices.
      - [gebsrmm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/gebsrmm/): Showcases a sparse matrix-matrix multiplication using GEBSR storage format.
      - [gemmi](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/gemmi/): Showcases a dense matrix sparse matrix multiplication using CSR storage format.
      - [sddmm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/sddmm/): Showcases a sampled dense-dense matrix multiplication using CSR storage format.
      - [spmm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/spmm/): Showcases a sparse matrix-dense matrix multiplication.
      - [spsm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_3/spsm/): Showcases a sparse triangular linear system solver using CSR storage format.
    - [preconditioner](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/preconditioner/): Manipulations on sparse matrices to obtain sparse preconditioner matrices.
      - [bsric0](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/preconditioner/bsric0/): Shows how to compute the incomplete Cholesky decomposition of a Hermitian positive-definite sparse BSR matrix.
      - [bsrilu0](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/preconditioner/bsrilu0/): Showcases how to obtain the incomplete LU decomposition of a sparse BSR square matrix.
      - [csric0](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/preconditioner/csric0/): Shows how to compute the incomplete Cholesky decomposition of a Hermitian positive-definite sparse CSR matrix.
      - [csrilu0](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/preconditioner/csrilu0/): Showcases how to obtain the incomplete LU decomposition of a sparse CSR square matrix.
      - [csritilu0](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/preconditioner/csritilu0/): Showcases how to obtain iteratively the incomplete LU decomposition of a sparse CSR square matrix.
      - [gpsv](https://github.com/amd/rocm-examples/tree/develop/Libraries/rocSPARSE/preconditioner/gpsv/): Shows how to compute the solution of pentadiagonal linear system.
      - [gtsv](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/preconditioner/gtsv/): Shows how to compute the solution of a tridiagonal linear system.
  - [rocThrust](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocThrust/)
    - [device_ptr](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocThrust/device_ptr/): Simple program that showcases the usage of the `thrust::device_ptr` template.
    - [norm](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocThrust/norm/): An example that computes the Euclidean norm of a `thrust::device_vector`.
    - [reduce_sum](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocThrust/reduce_sum/): An example that computes the sum of a `thrust::device_vector` integer vector using the `thrust::reduce()` generalized summation and the `thrust::plus` operator.
    - [remove_points](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocThrust/remove_points/): Simple program that demonstrates the usage of the `thrust` random number generation, host vector, generation, tuple, zip iterator, and conditional removal templates. It generates a number of random points in a unit square and then removes all of them outside the unit circle.
    - [saxpy](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocThrust/saxpy/): Simple program that implements the SAXPY operation (`y[i] = a * x[i] + y[i]`) using rocThrust and showcases the usage of the vector and functor templates and of `thrust::fill` and `thrust::transform` operations.
    - [vectors](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocThrust/vectors/): Simple program that showcases the `host_vector` and the `device_vector` of rocThrust.
- [Tutorials](https://github.com/ROCm/rocm-examples/tree/develop/Tutorials/): Showcases HIP Documentation Tutorials.
  - [reduction](https://github.com/ROCm/rocm-examples/tree/develop/Tutorials/reduction/): Showcases a reduction tutorial for HIP Documentation.
