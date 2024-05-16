# HIP-Basic Module API Example

## Description

This example shows how to load and execute a HIP module in runtime without linking it to the rest of the code during compilation.

### Application flow

1. Set up the name of the compiled module code object file `(*.co)`, located in the same directory.
2. Define kernel launch parameters.
3. Initialize input and output vectors in host memory.
4. Allocate arrays and copy the input and output vectors to the device memory.
5. Get the module path from the module file name.
6. Load module by `hipModuleLoad()`.
7. Fetch a reference to the kernel by `hipModuleGetFunction()`.
8. Create and fill the array with kernel arguments.
9. Launch the kernel on the default stream by `hipModuleLaunchKernel()`.
10. Copy the result back to the host.
11. Free input and output arrays on device memory.
12. Compare input and output vectors. The result of the comparison is printed to standard output.

## Building

The kernel module needs to be compiled as a non-linked device code object file (`*.co`), in one of the following ways:

- `hipcc --genco --offload-arch=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]`
- `clang++ --cuda-device-only --offload-arch=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]`

where the parameters are:

- `[TARGET GPU]`: GPU architecture (e.g. `gfx908` or `gfx90a:xnack-`).
- `[INPUT FILE]`: Name of the file containing kernels (e.g. `module.hip`).
- `[OUTPUT FILE]`: Name of the generated code object file (e.g. `module.co`).

The `main.hip` example file is compiled similarly as in the other examples.

## Key APIs and Concepts

- The `hipModuleLoad(hipModule_t *module, const char *file_name)` will load a HIP module in execution time from the path that is given as an input parameter or return an error.

- The `hipModuleGetFunction(hipFunction_t *kernel_function, hipModule_t module, const char *kernel_name)` will fetch a reference to the `__global__` kernel function in the HIP module.

- `hipModuleLaunchKernel` will launch kernel function on the device. The input parameters are:

  - `hipFunction_t kernel_function` Kernel function.
  - `unsigned int gridDimX`: Number of blocks in the dimension X.
  - `unsigned int gridDimY`: Number of blocks in the dimension Y.
  - `unsigned int gridDimZ`: Number of blocks in the dimension Z.
  - `unsigned int blockDimX`: Number of threads in the dimension X in a block.
  - `unsigned int blockDimY`: Number of threads in the dimension Y in a block.
  - `unsigned int blockDimZ`: Number of threads in the dimension Z in a block.
  - `unsigned int sharedMemBytes`: Amount of dynamic shared memory that will be available to each workgroup, in bytes. (Not used in this example.)
  - `hipStream_t stream`: The device stream, on which the kernel should be dispatched. (`hipStreamDefault` int this example.)
  - `void **kernelParams`: Pointer to the arguments needed by the kernel. Note that this parameter is not yet implemented, and thus the _extra_ parameter (the last one described in this list) should be used to pass arguments to the kernel. (Thereby `nullptr` is used in the example.)
  - `void **extra`: Pointer to all extra arguments passed to the kernel. They must be in the memory layout and alignment expected by the kernel. The list of arguments must end with `HIP_LAUNCH_PARAM_END`.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `__global__`
- `threadIdx`

#### Host symbols

- `hipGetLastError`
- `hipGetSymbolAddress`
- `hipGetSymbolSize`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
- `hipFree`
- `hipModuleLoad`
- `hipModuleGetFunction`
- `hipModuleLaunchKernel`
