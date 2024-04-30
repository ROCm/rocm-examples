// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "cmdparser.hpp"
#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ostream>
#include <random>
#include <vector>

/// \brief Calculates the prefix sum within a block, in place.
__global__ void block_prefix_sum(float* d_data, int size, int offset)
{
    const int thread_id  = threadIdx.x;
    const int block_id   = blockIdx.x;
    const int block_size = blockDim.x;

    const int x = (offset * (2 * (block_id * block_size + thread_id) + 1)) - 1;

    // Cache the computational window in shared memory
    extern __shared__ float block[];
    if(x < size)
    {
        block[2 * thread_id] = d_data[x];
    }
    if(x + offset < size)
    {
        block[2 * thread_id + 1] = d_data[x + offset];
    }

    // Build up tree
    int tree_offset = 1;
    for(int tree_size = size >> 1; tree_size > 0; tree_size >>= 1)
    {
        __syncthreads();
        if(thread_id < tree_size)
        {
            int from = tree_offset * (2 * thread_id + 1) - 1;
            int to   = tree_offset * (2 * thread_id + 2) - 1;
            block[to] += block[from];
        }
        tree_offset <<= 1;
    }

    if(size > 2)
    {
        if(tree_offset < size)
        {
            tree_offset <<= 1;
        }

        // Build down tree
        int max_thread = tree_offset >> 1;
        for(int tree_size = 0; tree_size < max_thread; tree_size <<= 1)
        {
            tree_size += 1;
            tree_offset >>= 1;
            __syncthreads();

            if(thread_id < tree_size)
            {
                int from = tree_offset * (thread_id + 1) - 1;
                int to   = from + (tree_offset >> 1);
                block[to] += block[from];
            }
        }
    }
    __syncthreads();

    // write the results back to global memory
    if(x < size)
    {
        d_data[x] = block[2 * thread_id];
    }
    if(x + offset < size)
    {
        d_data[x + offset] = block[2 * thread_id + 1];
    }
}

/// \brief Propogates values of the prefix sum between blocks on a device.
__global__ void device_prefix_sum(float* buffer, int size, int offset)
{
    const int thread_id  = threadIdx.x;
    const int block_size = blockDim.x;
    const int block_id   = blockIdx.x;

    const int sorted_blocks = offset / block_size;
    const int unsorted_block_id
        = block_id + (block_id / ((offset << 1) - sorted_blocks) + 1) * sorted_blocks;
    int x = (unsorted_block_id * block_size + thread_id);
    if(((x + 1) % offset != 0) && (x < size))
    {
        buffer[x] += buffer[x - (x % offset + 1)];
    }
}

void run_prefix_sum_kernels(float* input, float* output, const int size)
{
    // 4.1 Define kernel constants
    constexpr unsigned int threads_per_block = 128;
    dim3                   block_dim(threads_per_block);

    // Each thread works on 2 elements.
    constexpr unsigned int items_per_block = threads_per_block * 2;
    // block_prefix_sum uses shared memory dependent on the amount of threads per block.
    constexpr size_t shared_size = sizeof(float) * 2 * threads_per_block;

    // 4.2 Declare and allocate device memory.
    float* d_data;
    HIP_CHECK(hipMalloc(&d_data, sizeof(float) * size));

    // 4.3 Copy the inputs from host to device
    HIP_CHECK(hipMemcpy(d_data, input, sizeof(float) * size, hipMemcpyHostToDevice));

    // 4.4 Sweep over the input, multiple times if needed
    // Alternatively, use hipcub::DeviceScan::ExclusiveScan
    for(int offset = 1; offset < size; offset *= items_per_block)
    {
        const unsigned int data_size = size / offset;

        if(size / offset > 1)
        {
            unsigned int total_threads = (data_size + 1) / 2;
            total_threads = ceiling_div(total_threads, threads_per_block) * threads_per_block;
            dim3 grid_dim(total_threads / threads_per_block);

            block_prefix_sum<<<grid_dim, block_dim, shared_size>>>(d_data, size, offset);
        }

        if(offset > 1)
        {
            unsigned int total_threads = size - offset;
            total_threads -= (total_threads / (offset * items_per_block)) * offset;
            total_threads = ceiling_div(total_threads, threads_per_block) * threads_per_block;
            dim3 grid_dim(total_threads / threads_per_block);

            device_prefix_sum<<<grid_dim, block_dim>>>(d_data, size, offset);
        }
    }

    // 4.5 Copy the results from device to host.
    HIP_CHECK(hipMemcpy(output, d_data, sizeof(float) * size, hipMemcpyDeviceToHost));

    // 4.6 Clean up device memory allocations.
    HIP_CHECK(hipFree(d_data));
}

int main(int argc, char* argv[])
{
    // 1. Parse user input.
    cli::Parser parser(argc, argv);
    parser.set_optional("n", "size", 256);
    parser.run_and_exit_if_error();

    const int size = parser.get<int>("n");
    if(size <= 0)
    {
        std::cout << "Size must be at least 1." << std::endl;
        exit(0);
    }

    // 2. Generate input vector.
    std::cout << "Prefix sum over " << size << " items.\n" << std::endl;

    std::vector<float> input(size);
    std::vector<float> output(size);

    std::default_random_engine            generator;
    std::uniform_real_distribution<float> distribution(-1, 1);

    std::generate(input.begin(), input.end(), [&]() { return distribution(generator); });

    // 3. Run the prefix sum.
    run_prefix_sum_kernels(input.data(), output.data(), size);

    // 4. Verify the output.
    float verify = 0;
    int   errors = 0;
    for(int i = 0; i < size; i++)
    {
        verify += input[i];
        errors += std::pow(output[i] - verify, 2) > 1e-8;
    }

    std::cout << "Final sum on \n"
              << "  device: " << output.back() << "\n"
              << "  host  : " << verify << "\n"
              << std::endl;

    return report_validation_result(errors);
}
