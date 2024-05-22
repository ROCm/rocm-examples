# HIP-Basic Runtime Compilation Example

## Description

Runtime compilation allows compiling fragments of source code to machine code at runtime, when a program is already running, rather than compiling the code ahead of time. HIP supports runtime compilation through hipRTC, which can be used to compile HIP device code at runtime. This permits specific optimizations that depend on values determined at runtime. Therefore, usage of hipRTC provides the possibility of obtaining optimizations and performance improvements over offline compilation.

This example showcases how to make use of hipRTC to compile in runtime a kernel and launch it on a device. This kernel is a simple SAXPY, i.e. a single-precision operation $y_i=ax_i+y_i$.

### Application flow

The diagram below summarizes the runtime compilation part of the example.

1. A number of variables are declared and defined to configure the program which will be compiled in runtime.
2. The program is created using the above variables as parameters, along with the SAXPY kernel in string form.
3. The properties of the first device (GPU) available are consulted to set the device architecture as (the only) compile option.
4. The program is compiled using the previously mentioned compile options.
5. If exists, the log generated during the compile process is printed to the standard output.
6. The binary compiled from the program is stored as a vector of characters and the program object is destroyed.
7. Begin the preparation for the launch of the kernel on the device. A number of constants are defined to control the problem details and the kernel launch parameters.
8. The two input vectors, $x$ and $y$, are instantiated in host memory and filled with the increasing sequences $1, 2, 3, 4, ...$ and $2, 4, 6, 8, ...$, respectively.
9. The necessary amount of device (GPU) memory is allocated and the elements of the input vectors are copied to the device memory.
10. A HIP module corresponding to the compiled binary is loaded into the current context and the SAXPY kernel is extracted from it into a HIP function object.
11. The kernel launch configuration options and its arguments are declared and defined.
12. A trace message is printed to the standard output.
13. The GPU kernel is then launched with the above mentioned options along with the constants defined previously.
14. The results are copied back to host vector $y$.
15. The previously allocated device memory is freed.
16. The module is unloaded from the current context and freed.
17. The first few elements of the result vector $y$ are printed to the standard output.

![A diagram to visualize the runtime compilation and launch of this example](hiprtc.svg)

## Key APIs and Concepts

- `hipGetDeviceProperties` extracts the properties of the desired device. In this example it is used to get the GPU architecture.
- `hipModuleGetFunction` extracts a handle for a function with a certain name from a given module. Note that if no function with that name is present in the module this method will return an error.
- `hipModuleLaunchKernel` queues the launch of the provided kernel on the device. This function normally presents an asynchronous behaviour (see `HIP_LAUNCH_BLOCKING`), i.e. a call to it may return before the device finishes the execution of the kernel. Its parameters are the following:

  - The kernel to be launched.
  - Number of blocks in the dimension X of kernel grid, i.e. the X component of grid size.
  - Number of blocks in the dimension Y of kernel grid, i.e. the Y component of grid size.
  - Number of blocks in the dimension Z of kernel grid, i.e. the Z component of grid size.
  - Number of threads in the dimension X of each block, i.e. the X component of block size.
  - Number of threads in the dimension Y of each block, i.e. the Y component of block size.
  - Number of threads in the dimension Z of each block, i.e. the Z component of block size.
  - Amount of dynamic shared memory that will be available to each workgroup, in bytes. Not used in this example.
  - The device stream, on which the kernel should be dispatched. If 0 (or NULL), the NULL stream will be used. In this example the latter is used.
  - Pointer to the arguments needed by the kernel. Note that this parameter is not yet implemented, and thus the _extra_ parameter (the last one described in this list) should be used to pass arguments to the kernel.
  - Pointer to all extra arguments passed to the kernel. They must be in the memory layout and alignment expected by the kernel. The list of arguments must end with `HIP_LAUNCH_PARAM_END`.

- `hipModuleLoadData` builds a module from a code (compiled binary) object residing in host memory and loads it into the current context. Note that in this example this function is called right after `hipMalloc`. This is due to the fact that, on CUDA, `hipModuleLoadData` will fail if it is not called after some runtime API call is done (as it will implicitly intialize a current context) or if there is not an explicit creation of a (current) context.
- `hipModuleUnload` unloads the specified module from the current context and frees it.
- `hiprtcCompileProgram` compiles the given program in runtime. Some compilation options may be passed as parameters to this function. In this example, the GPU architeture is the only compilation option.
- `hiprtcCreateProgram` instantiates a runtime compilation program from the given parameters. Those are the following:

  - The runtime compilation program object that will be set with the new instance.
  - A pointer to the program source code.
  - A pointer to the program name.
  - The number of headers to be included.
  - An array of pointers to the headers names.
  - An array of pointers to the names to be included in the source program.

    In this example the program is created including two header files to illustrate how to pass all of the above arguments to this function.

- `hiprtcDestroyProgram` destroys an instance of a given runtime compilation program object.
- `hiprtcGetProgramLog` extracts the char pointer to the log generated during the compilation of a given runtime compilation program.
- `hiprtcGetProgramLogSize` returns the compilation log size of a given runtime compilation program, measured as number of characters.
- `hiprtcGetCode` extracts the char pointer to the compilation binary in memory from a runtime compilation program object. This binary is needed to load the corresponding HIP module into the current context and extract from it the kernel(s) that will be executed on the GPU.
- `hiprtcGetCodeSize` returns the size of the binary compiled of a given runtime compilation program, measured as number of characters.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`

#### Host symbols

- `hipFree`
- `hipGetDeviceProperties`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipModuleGetFunction`
- `hipModuleLaunchKernel`
- `hipModuleLoadData`
- `hipModuleUnload`
- `hiprtcCompileProgram`
- `hiprtcCreateProgram`
- `hiprtcDestroyProgram`
- `hiprtcGetCode`
- `hiprtcGetCodeSize`
- `hiprtcGetProgramLog`
- `hiprtcGetProgramLogSize`
- `HIP_LAUNCH_PARAM_BUFFER_POINTER`
- `HIP_LAUNCH_PARAM_BUFFER_SIZE`
- `HIP_LAUNCH_PARAM_END`
