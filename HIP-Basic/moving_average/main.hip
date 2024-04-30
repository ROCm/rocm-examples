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

/// \brief Compute the moving average of \p input_size elements with a window size of \p WindowSize.
/// Shared memory is used to cache the values needed by all threads in the block.
/// Thread \p i computes the average of values <tt>[i, i + WindowSize)</tt>.
template<unsigned int BlockSize, unsigned int WindowSize>
__global__ void
    moving_average(const unsigned int* input, unsigned int* output, const unsigned int input_size)
{
    // The offset of this block from the start of the grid.
    const unsigned int block_offset = blockIdx.x * blockDim.x;
    // The index of this thread in the grid.
    const unsigned int thread_idx = block_offset + threadIdx.x;

    // The number of values needed to compute BlockSize averages.
    constexpr unsigned int  buffer_size = BlockSize + WindowSize - 1;
    __shared__ unsigned int buffer[buffer_size];

    // Load values into shared memory.
    // Note that threadIdx.x is the index in the block.
    for(unsigned int i = threadIdx.x; i < buffer_size; i += BlockSize)
    {
        // Index in the input array.
        const unsigned int index = block_offset + i;
        // Prevent reading outside the input array.
        if(index < input_size)
        {
            buffer[i] = input[index];
        }
    }

    // Wait for all threads to load into shared memory.
    __syncthreads();

    // Compute the average using the cached data.
    const unsigned int output_size = input_size - WindowSize + 1;
    if(thread_idx < output_size)
    {
        unsigned int sum = 0;
        for(unsigned int i = 0; i < WindowSize; i++)
        {
            sum += buffer[threadIdx.x + i];
        }

        sum /= WindowSize;

        output[thread_idx] = sum;
    }
}

int main()
{
    // The number of elements to process.
    constexpr unsigned int input_size = 10000000;

    // The number of elements to compute the average over.
    constexpr unsigned int window_size = 97;
    static_assert(input_size >= window_size);

    // The number of moving average values produced.
    constexpr unsigned int output_size = input_size - window_size + 1;

    // The number of threads per kernel block.
    constexpr unsigned int block_size = 256;

    // Number of blocks per kernel grid.
    constexpr unsigned int grid_size = ceiling_div(output_size, block_size);

    // Allocate and initialize input data on the host.
    std::vector<unsigned int> h_input(input_size);
    std::iota(h_input.begin(), h_input.end(), 0);
    std::transform(h_input.begin(),
                   h_input.end(),
                   h_input.begin(),
                   [&](unsigned int i) { return i % window_size; });

    // Allocate device input data and copy host data to it.
    unsigned int*    d_input{};
    constexpr size_t input_size_bytes = input_size * sizeof(unsigned int);
    HIP_CHECK(hipMalloc(&d_input, input_size_bytes));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), input_size_bytes, hipMemcpyHostToDevice));

    // Allocate device output data.
    unsigned int*    d_output{};
    constexpr size_t output_size_bytes = output_size * sizeof(unsigned int);
    HIP_CHECK(hipMalloc(&d_output, output_size_bytes));

    std::cout << "Calculating the moving average of " << input_size << " elements with window size "
              << window_size << std::endl;

    // Launch the kernel on the default stream.
    moving_average<block_size, window_size>
        <<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(d_input, d_output, input_size);

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
    std::vector<unsigned int> h_output(output_size);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, output_size_bytes, hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_input));

    // Validate output.
    unsigned int incorrect_count = 0;
    for(unsigned int i = 0; i < output_size; i++)
    {
        // Average of [0, n) is given by (n - 1) / 2.
        incorrect_count += (h_output[i] != (window_size - 1) / 2);
    }

    if(incorrect_count)
    {
        std::cout << "Validation failed. Errors: " << incorrect_count << std::endl;
        return error_exit_code;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }
}
