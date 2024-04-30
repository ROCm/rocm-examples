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

/// \brief A simple kernel that copies each value from the input to the output.
__global__ void copy_kernel(uint32_t* out, const uint32_t* in, const unsigned int size)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size)
    {
        out[tid] = in[tid];
    }
}

/// \brief The main entry point of this static host library: This function is exported
/// from this library, and can be called from other libraries or executables if this library
/// is linked with it.
///
/// It performs a simple test invocation of a HIP kernel. If the test passes a message is printed
/// and the function returns 0. Otherwise an error message is printed and the function returns
/// `error_exit_code`.
int run_test()
{
    // The number of elements in the input vector.
    constexpr unsigned int size = 512;

    // The number of bytes to allocate for the input- and output device vectors.
    constexpr size_t size_bytes = size * sizeof(uint32_t);

    // The number of threads per kernel block.
    constexpr unsigned int block_size = 256;

    // The number of blocks per kernel grid. The expression below calculates `ceil(size / block_size)`.
    constexpr unsigned int grid_size = ceiling_div(size, block_size);

    // Allocate host input vector and fill it with an increasing sequence (i.e. 0, 1, 2, ...).
    std::vector<uint32_t> in(size);
    std::iota(in.begin(), in.end(), 0);

    // Allocate input and output device vector and copy the input data.
    uint32_t* d_in{};
    uint32_t* d_out{};
    HIP_CHECK(hipMalloc(&d_in, size_bytes));
    HIP_CHECK(hipMalloc(&d_out, size_bytes));
    HIP_CHECK(hipMemcpy(d_in, in.data(), size_bytes, hipMemcpyHostToDevice));

    // Launch the kernel on the default stream.
    copy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(d_out, d_in, size);

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
    std::vector<uint32_t> out(size);
    HIP_CHECK(hipMemcpy(out.data(), d_out, size_bytes, hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    // Check the results' validity.
    size_t errors = 0;
    for(size_t i = 0; i < size; ++i)
    {
        if(in[i] != out[i])
        {
            ++errors;
        }
    }

    if(errors != 0)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        // Return control flow to the main program.
        return error_exit_code;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }

    // Return control flow to the main program.
    return 0;
}
