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

#include <iostream>
#include <numeric>
#include <vector>

#if __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#else
    static_assert(false, "filesystem not available");
#endif

int main(int, char* argv[])
{
    // The module file that contains the kernel that we want to invoke. This
    // file is expected to be in the same directory as the executable.
    constexpr const char* module_file_name = "module.co";

    // The size of the input and output vectors.
    constexpr unsigned int size = 64;

    // The total number of bytes in the input and output vectors.
    constexpr size_t size_bytes = size * sizeof(float);

    // Number of threads per kernel block.
    constexpr unsigned int block_size = size;

    // Number of blocks per kernel grid. The expression below calculates ceil(size/block_size).
    constexpr unsigned int grid_size = ceiling_div(size, block_size);

    // Allocate host vectors for the input and output.
    std::vector<float> h_in(size);
    std::vector<float> h_out(size);

    // Fill the input with an increasing sequence (i.e. 1, 2, 3, 4...).
    std::iota(h_in.begin(), h_in.end(), 1.f);

    // Allocate and copy vectors to device memory.
    float* d_in{};
    float* d_out{};
    HIP_CHECK(hipMalloc(&d_in, size_bytes));
    HIP_CHECK(hipMalloc(&d_out, size_bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), size_bytes, hipMemcpyHostToDevice));

    // Compute an absolute path to the module that we are going to load.
    // To do that, find the directory where the example executable is placed in from the 0th argument.
    // Note that this does not always work (the executable may be invoked with a completely different
    // value for argv[0]), but works for the purposes of this example.
    fs::path exe_dir
        = fs::canonical(fs::path(argv[0])).parent_path();
    fs::path module_path = exe_dir / module_file_name;

    // Load the module from the path that we just constructed.
    // If the module does not exist, this function will return an error.
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, module_path.u8string().c_str()));

    // Fetch a reference to the kernel that we are going to invoke.
    hipFunction_t kernel;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "test_module_api_kernel"));

    // Create and fill array with kernel arguments.
    size_t offset    = 0;
    char   args[256] = {};

    *(reinterpret_cast<float**>(&args[offset])) = d_out;
    offset += sizeof(d_out);
    *(reinterpret_cast<float**>(&args[offset])) = d_in;
    offset += sizeof(d_in);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &offset,
                      HIP_LAUNCH_PARAM_END};

    // Launch the kernel on the default stream and with the above configuration.
    HIP_CHECK(hipModuleLaunchKernel(kernel,
                                    grid_size,
                                    1,
                                    1,
                                    block_size,
                                    1,
                                    1,
                                    0,
                                    hipStreamDefault,
                                    nullptr,
                                    config));

    // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, size_bytes, hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    // Check the results' validity.
    unsigned int errors{};
    for(size_t i = 0; i < size; ++i)
    {
        if(h_out[i] != h_in[i])
        {
            ++errors;
        }
    }

    if(errors != 0)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        return error_exit_code;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }

    return 0;
}
