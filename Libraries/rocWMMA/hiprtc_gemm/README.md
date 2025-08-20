# rocWMMA hipRTC General Matrix Multiplication (GEMM)

## Description

This example demonstrates runtime compilation of rocWMMA kernels using hipRTC (HIP Runtime Compilation). It shows how to compile and execute WMMA kernels dynamically at runtime, providing flexibility for applications that need to generate optimized kernels based on runtime parameters or conditions.

The operation calculates the following product:

$D = \alpha \cdot A \cdot B + \beta \cdot C$

where:

- $\alpha$ and $\beta$ are scalars
- $A$ is a matrix of dimensions $m \times k$
- $B$ is a matrix of dimensions $k \times n$
- $C$ and $D$ are matrices of dimensions $m \times n$

## Application flow

1. **Set up ROCm Path**: The ROCm installation path is determined from environment variables or a default location to locate the rocWMMA header files.
2. **Define Kernel Source**: The rocWMMA GEMM kernel is defined as a string literal within the host code.
3. **Compile Kernel with hipRTC**:
    - A hipRTC program is created from the kernel source string.
    - The program is compiled using `hiprtcCompileProgram` with necessary compiler options, including the path to rocWMMA headers.
    - Error handling is implemented to catch and report any compilation failures.
4. **Load Compiled Module**:
    - The compiled kernel code is retrieved from the hipRTC program.
    - A HIP module is created from the compiled code using `hipModuleLoadData`.
    - The GEMM kernel function is extracted from the module using `hipModuleGetFunction`.
5. **Set up Matrices**:
    - Matrix dimensions are defined.
    - Host-side memory for matrices A, B, C, and D is allocated and initialized. Matrix A and B are initialized with random values, and C is initialized to zero.
6. **Manage Device Memory**:
    - Device memory is allocated for matrices A, B, C, and D.
    - Input matrices (A, B, and C) are copied from host to device.
7. **Execute Kernel**:
    - Kernel launch parameters (grid and block dimensions) are configured.
    - A struct containing kernel arguments (matrix pointers, dimensions, scalars) is created.
    - The kernel is launched using `hipModuleLaunchKernel`.
8. **Retrieve and Validate Results**:
    - The resulting matrix D is copied from device to host.
    - The result is validated by comparing it against a CPU-based reference GEMM implementation.
9. **Clean Up**: All hipRTC resources, HIP modules, and device memory are released.

## Key APIs and Concepts

- **hipRTC Program Management**:
  - `hiprtcCreateProgram()`: Creates a program object from a string of source code.
  - `hiprtcCompileProgram()`: Compiles the program with specified options.
  - `hiprtcGetCode()` and `hiprtcGetCodeSize()`: Retrieve the compiled machine code (cubin).
  - `hiprtcGetProgramLog()` and `hiprtcGetProgramLogSize()`: Get compilation logs, useful for debugging.
  - `hiprtcDestroyProgram()`: Frees the program object.

- **HIP Module Management**:
  - `hipModuleLoadData()`: Loads the compiled code into a HIP module.
  - `hipModuleGetFunction()`: Gets a handle to a specific kernel function within the module.
  - `hipModuleLaunchKernel()`: Launches the kernel with specified parameters.
  - `hipModuleUnload()`: Unloads the module.

- **rocWMMA Fragments**: In the runtime-compiled kernel, rocWMMA fragments (`rocwmma::fragment`) are used to represent portions of matrices that are processed by each thread. This is the core of the rocWMMA API.

- **rocWMMA Intrinsics**:
  - `rocwmma::load_matrix_sync()`: Loads data from global memory into fragments.
  - `rocwmma::mma_sync()`: Performs the matrix multiplication and accumulation on fragments.
  - `rocwmma::store_matrix_sync()`: Stores data from fragments back to global memory.
  - `rocwmma::fill_fragment()`: Initializes a fragment with a specific value.

## Demonstrated API Calls

### hipRTC

- `hiprtcCreateProgram`
- `hiprtcCompileProgram`
- `hiprtcGetCodeSize`
- `hiprtcGetCode`
- `hiprtcGetProgramLog`
- `hiprtcGetProgramLogSize`
- `hiprtcDestroyProgram`
- `hiprtcGetErrorString`

### HIP Module API

- `hipModuleLoadData`
- `hipModuleGetFunction`
- `hipModuleLaunchKernel`
- `hipModuleUnload`

### rocWMMA (in runtime-compiled kernel)

- `rocwmma::fragment`
- `rocwmma::load_matrix_sync`
- `rocwmma::store_matrix_sync`
- `rocwmma::mma_sync`
- `rocwmma::fill_fragment`

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
- `hipGetDevice`
- `hipGetDeviceProperties`
- `hipEventCreate`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipEventElapsedTime`
- `hipEventDestroy`

## Data Types and Enums

- `hiprtcProgram`
- `hipModule_t`
- `hipFunction_t`
- `hipDeviceptr_t`
- `rocwmma::bfloat16_t`
- `rocwmma::row_major`
- `rocwmma::col_major`
