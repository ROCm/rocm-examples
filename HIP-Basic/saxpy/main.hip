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

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <cstddef>

/// \brief Calculates \p d_y[i]=a*d_x[i]+d_y[i] where \p i stands for the thread's index in the grid.
__global__ void saxpy_kernel(const float a, const float* d_x, float* d_y, const unsigned int size)
{
    // Compute the current thread's index in the grid.
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // The grid can be larger than the number of items in the vectors. Avoid out-of-bounds addressing.
    if(global_idx < size)
    {
        d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
    }
}

int main()
{
    // The number of float elements in each vector.
    constexpr unsigned int size = 1000000;

    // Bytes to allocate for each device vector.
    constexpr size_t size_bytes = size * sizeof(float);

    // Number of threads per kernel block.
    constexpr unsigned int block_size = 256;

    // Number of blocks per kernel grid. The expression below calculates ceil(size/block_size).
    constexpr unsigned int grid_size = ceiling_div(size, block_size);

    // The constant value to use in the a*x+y formula.
    constexpr float a = 2.f;

    // Allocate x vector and fill it with an increasing sequence (i.e. 1, 2, 3, 4...)
    std::vector<float> x(size);
    std::iota(x.begin(), x.end(), 1.f);

    // Allocate y vector and fill it with a constant of 1.
    std::vector<float> y(size);
    std::fill(y.begin(), y.end(), 1.f);

    // Allocate and copy vectors to device memory.
    float* d_x{};
    float* d_y{};
    HIP_CHECK(hipMalloc(&d_x, size_bytes));
    HIP_CHECK(hipMalloc(&d_y, size_bytes));
    HIP_CHECK(hipMemcpy(d_x, x.data(), size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, y.data(), size_bytes, hipMemcpyHostToDevice));

    std::cout << "Calculating y[i] = a * x[i] + y[i] over " << size << " elements." << std::endl;

    // Launch the kernel on the default stream.
    saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(a, d_x, d_y, size);

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
    HIP_CHECK(hipMemcpy(y.data(), d_y, size_bytes, hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    // Print the first few elements of the results:
    constexpr size_t elements_to_print = 10;
    std::cout << "First " << elements_to_print << " elements of the results: "
              << format_range(y.begin(), y.begin() + elements_to_print) << std::endl;
}
