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

#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

constexpr unsigned int device_array_size = 16;

/// A test global variable of a single element, that will later be set from the host.
__device__ float global;

/// A test global variable of \p device_array_size elements that will be set from the host.
__device__ float global_array[device_array_size];

/// \brief A simple test kernel, that reads from <tt>in</tt>, <tt>global</tt>, and
/// <tt>global_array</tt>. The result will be written to <tt>out</tt>.
__global__ void test_globals_kernel(float* out, const float* in, const size_t size)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size)
    {
        out[tid] = in[tid] + global + global_array[tid % device_array_size];
    }
}

/// \brief Computes a reference result on the host, that is (if everything goes well)
/// hopefully equal to the results from the \p test_globals_kernel kernel.
std::vector<float> test_globals_reference(const std::vector<float>& in,
                                          const std::vector<float>  global_array,
                                          const float               global)
{
    std::vector<float> out(in.size());
    for(size_t i = 0; i < in.size(); ++i)
    {
        out[i] = in[i] + global + global_array[i % global_array.size()];
    }

    return out;
}

int main()
{
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

    // Fetch a device pointer to the device variable "global". We can pass the relevant
    // symbol directly to this function.
    void*  d_global{};
    size_t global_size_bytes{};
    HIP_CHECK(hipGetSymbolAddress(&d_global, HIP_SYMBOL(global)));
    HIP_CHECK(hipGetSymbolSize(&global_size_bytes, HIP_SYMBOL(global)));
    assert(global_size_bytes == sizeof(float));

    // This pointer is a regular device pointer, and so we may use it in the same ways
    // as pointers allocated using `hipMalloc`.
    constexpr float h_global = 42.f;
    HIP_CHECK(hipMemcpy(d_global, &h_global, global_size_bytes, hipMemcpyHostToDevice));

    // Set up the inputs for `global_array`.
    std::vector<float> h_global_array(device_array_size);
    for(size_t i = 0; i < h_global_array.size(); ++i)
    {
        h_global_array[i] = i * 1000.f;
    }

    // Initialize `global_array` by copying to it directly, omitting the need to fetch it first.
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(global_array),
                                h_global_array.data(),
                                h_global_array.size() * sizeof(float)));

    // Launch the kernel on the default stream and with the above configuration.
    test_globals_kernel<<<dim3(block_size), dim3(grid_size), 0, hipStreamDefault>>>(d_out,
                                                                                    d_in,
                                                                                    size);

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, size_bytes, hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    // Compute the expected values on the host.
    const std::vector<float> reference = test_globals_reference(h_in, h_global_array, h_global);

    // Check the results' validity.
    constexpr float eps = 1.0E-6f;
    unsigned int    errors{};
    for(size_t i = 0; i < size; ++i)
    {
        if(std::fabs(h_out[i] - reference[i]) > eps)
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
