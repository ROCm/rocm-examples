// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "example_utils.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

// SAXPY kernel stored as a string
static constexpr auto saxpy_kernel{
    // MSVC 19.16 does not properly handle R-strings in its preprocessor when they are on a separate line,
    // if the /E flag is passed (as NVCC does).
    "#include \"test_header.h\"\n"
    "#include \"test_header1.h\"\n"
    // MSVC 19.16 does not properly preprocess extern "C" when it's not on the same line as the
    // function declaration if the /E flag is passed (as NVCC does).
    R"(
extern "C" __global__ void saxpy_kernel(const real a, const realptr d_x, realptr d_y, const unsigned int size)
{
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx < size)
    {
        d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
    }
}
)"};

int main()
{
    // Program to be compiled in runtime.
    hiprtcProgram prog;

    // Vector containing example header names.
    std::vector<const char*> header_names;
    header_names.push_back("test_header.h");
    header_names.push_back("test_header1.h");

    // Vector containing example names to be included in the program.
    std::vector<const char*> header_sources;
    header_sources.push_back("#ifndef HIPRTC_TEST_HEADER_H\n#define HIPRTC_TEST_HEADER_H\ntypedef "
                             "float real;\n#endif //HIPRTC_TEST_HEADER_H\n");
    header_sources.push_back(
        "#ifndef HIPRTC_TEST_HEADER1_H\n#define HIPRTC_TEST_HEADER1_H\ntypedef float* "
        "realptr;\n#endif //HIPRTC_TEST_HEADER1_H\n");

    // Create program.
    hiprtcCreateProgram(&prog,
                        saxpy_kernel,
                        "saxpy_kernel.cu",
                        header_sources.size(),
                        header_sources.data(),
                        header_names.data());

    // Get device properties from the first device available.
    hipDeviceProp_t        props;
    constexpr unsigned int device_id = 0;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    std::vector<const char*> options;

    // Obtain architecture's name from device properties and initialize array of compile options. When in CUDA we omit this option.
#ifdef __HIP_PLATFORM_AMD__
    std::string arch_option;
    if(props.gcnArchName[0])
    {
        arch_option = std::string("--gpu-architecture=") + props.gcnArchName;
        options.push_back(arch_option.c_str());
    }
#endif

    // Compile program in runtime. Parameters are the program, number of options and array with options.
    const hiprtcResult compile_result{hiprtcCompileProgram(prog, options.size(), options.data())};

    // Get the size of the log (possibly) generated during the compilation.
    size_t log_size;
    hiprtcGetProgramLogSize(prog, &log_size);

    // If the compilation generated a log, print it.
    if(log_size)
    {
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        std::cout << log << std::endl;
    }

    // If the compilation failed, say so and exit.
    if(compile_result != HIPRTC_SUCCESS)
    {
        std::cout << "Error: compilation failed." << std::endl;
        return EXIT_FAILURE;
    }

    // Get the size (in number of characters) of the binary compiled from the program.
    size_t code_size;
    hiprtcGetCodeSize(prog, &code_size);

    // Store compiled binary as a vector of characters.
    std::vector<char> code(code_size);
    hiprtcGetCode(prog, code.data());

    // Destroy program object.
    hiprtcDestroyProgram(&prog);

    // Now we launch the kernel on the device.

    // Total number of float elements in each device vector.
    constexpr unsigned int size = 4096;

    // Total number of bytes to allocate for each device vector.
    constexpr size_t size_bytes = size * sizeof(float);

    // Number of threads per kernel block.
    constexpr unsigned int block_size = 128;

    // Number of blocks per kernel grid, calculated as ceil(size/block_size).
    constexpr unsigned int grid_size = ceiling_div(size, block_size);

    // Constant value 'a' to be used in the expression 'a*x+y'.
    constexpr float a = 5.1f;

    // Allocate x vector in host and fill it with increasing sequence 1, 2, 3, 4, ... .
    std::vector<float> x(size);
    std::iota(x.begin(), x.end(), 1.f);

    // Allocate y vector in host and fill it with increasing sequence 2, 4, 6, 8, ... .
    std::vector<float> y(x);
    std::for_each(y.begin(), y.end(), [](float& f) { f = 2 * f; });

    // Allocate vectors in device and copy from host to device memory.
    float* d_x{};
    float* d_y{};
    HIP_CHECK(hipMalloc(&d_x, size_bytes));
    HIP_CHECK(hipMalloc(&d_y, size_bytes));
    HIP_CHECK(hipMemcpy(d_x, x.data(), size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, y.data(), size_bytes, hipMemcpyHostToDevice));

    // Load the HIP module corresponding to the compiled binary into the current context.
    hipModule_t module;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));

    // Extract SAXPY kernel from module into a function object.
    hipFunction_t kernel;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "saxpy_kernel"));

    // Create and fill array with kernel arguments.
    size_t offset    = 0;
    char   args[256] = {};

    *(reinterpret_cast<float*>(&args[offset])) = a;
    offset += sizeof(a);
    offset += 4; // aligning fix for CUDA executions
    *(reinterpret_cast<float**>(&args[offset])) = d_x;
    offset += sizeof(d_x);
    *(reinterpret_cast<float**>(&args[offset])) = d_y;
    offset += sizeof(d_y);
    *(reinterpret_cast<unsigned int*>(&args[offset])) = size;
    offset += sizeof(size);

    // Create array with kernel arguments and its size.
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &offset,
                      HIP_LAUNCH_PARAM_END};

    std::cout << "Calculating y[i] = a * x[i] + y[i] over " << size << " elements." << std::endl;

    // Launch the kernel on the NULL stream and with the above configuration.
    HIP_CHECK(hipModuleLaunchKernel(kernel,
                                    grid_size,
                                    1,
                                    1,
                                    block_size,
                                    1,
                                    1,
                                    0,
                                    nullptr,
                                    nullptr,
                                    (void**)&config));

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError())

    // Copy results from device to host.
    HIP_CHECK(hipMemcpy(y.data(), d_y, size_bytes, hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    // Unload module.
    HIP_CHECK(hipModuleUnload(module));

    // Print the first few elements of the results for validation.
    constexpr size_t elements_to_print = 10;
    std::cout << "First " << elements_to_print << " elements of the results: "
              << format_range(y.begin(), y.begin() + elements_to_print) << std::endl;

    return 0;
}
