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

#include <cstdlib>
#include <iostream>
#include <vector>

/// \brief Device function to square each element
/// in the array `in` and write to array `out`.
template<typename T>
__global__ void vector_square_kernel(T* out, const T* in, const unsigned long long size)
{
    // Get the unique global thread ID
    const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread hops stride amount of elements to find the next
    // element to square
    const size_t stride = blockDim.x * gridDim.x;

    for(size_t i = offset; i < size; i += stride)
    {
        out[i] = in[i] * in[i];
    }
}

int main()
{
    // Set the problem size
    constexpr size_t size          = 1000000;
    constexpr size_t size_in_bytes = size * sizeof(float);

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0 /*deviceID*/));
    std::cout << "info: running on device " << props.name << "\n";

    std::cout << "info: allocate host mem (" << 2 * size_in_bytes / 1024.0 / 1024.0 << " MiB) "
              << "\n";

    // Declare the host side arrays
    std::vector<float> h_in(size);
    std::vector<float> h_out(size);

    // Initialize the host size input
    for(size_t i = 0; i < size; i++)
    {
        h_in[i] = 1.618f + i;
    }

    // Declare the device side arrays
    float *d_in, *d_out;
    std::cout << "info: allocate device mem (" << 2 * size_in_bytes / 1024.0 / 1024.0 << " MiB)\n";
    // Allocate the device side memory
    HIP_CHECK(hipMalloc(&d_in, size_in_bytes));
    HIP_CHECK(hipMalloc(&d_out, size_in_bytes));

    std::cout << "info: copy Host2Device\n";

    // Copy the input from host to the GPU device
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), size_in_bytes, hipMemcpyHostToDevice));

    // Set the number of blocks per kernel grid.
    constexpr unsigned int grid_size = 512;
    // Set the number of threads per kernel block.
    constexpr unsigned int threads_per_block = 256;

    std::cout << "info: launch 'vector_square_kernel' kernel\n";
    vector_square_kernel<<<grid_size, threads_per_block, 0, hipStreamDefault>>>(d_out, d_in, size);

    // Check that the kernel invocation was successful.
    HIP_CHECK(hipGetLastError());

    std::cout << "info: copy Device2Host\n";
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, size_in_bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    std::cout << "info: check result\n";
    for(size_t i = 0; i < size; i++)
    {
        if(h_out[i] != h_in[i] * h_in[i])
        {
            std::cerr << "FAILED! h_out[" << i << "] = " << h_out[i]
                      << ", expected:  " << h_in[i] * h_in[i] << '\n';
            exit(error_exit_code);
        }
    }
    std::cout << "PASSED!\n";
}
