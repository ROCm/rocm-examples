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

#include <iostream>
#include <numeric>

// HIP and rocPRIM API
#include <hip/hip_runtime.h>
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_reduce.hpp>

/// \brief Compute the sum of an array on the host CPU
std::vector<int> reduce_sum_host(const std::vector<int>& data,
                                 const unsigned int      run_size,
                                 const unsigned int      grid_size)
{
    std::vector<int> sum_per_block(grid_size);
    for(unsigned int i = 0; i < grid_size; i++)
    {
        const auto begin = data.begin() + i * run_size;
        const auto end   = data.begin() + std::min(data.size(), (size_t(1) + i) * run_size);
        sum_per_block[i] = std::accumulate(begin, end, 0);
    }
    return sum_per_block;
}

/// \brief `reduce_sum_kernel` performs a reduction on partitions of the input array using
/// the binary operator `BinaryOp`. The array is split up into partitions of size `BlockSize`
/// and each partition is reduced by a single block. For each partition the resulting scalar value
/// is stored in `device_output` on the device.
template<unsigned int BlockSize, unsigned int ItemsPerThread>
__global__ void
    reduce_sum_kernel(const int* device_input, int* device_output, const unsigned int input_size)
{
    using block_load   = rocprim::block_load<int, BlockSize, ItemsPerThread>;
    using block_reduce = rocprim::block_reduce<int, BlockSize>;

    // Get the offset for each thread block. This is where each thread block
    // will read the global input such as (device_input + items_in_previous_blocks).
    const unsigned int items_in_previous_blocks = blockIdx.x * BlockSize * ItemsPerThread;

    // Guard the size of elements that each block loads.
    unsigned int valid_items = BlockSize * ItemsPerThread;

    // Check if this thread block is the last, and set valid_items if it is.
    // This is to make sure that the last thread block does not overflow in case the
    // size of the global input is not divisible by the number of items per block.
    if(blockIdx.x == (gridDim.x - 1))
    {
        valid_items = input_size % valid_items;
    }

    // Load the corresponding input values from global memory
    // for all participatory threads. Store the value in a local thread variable.
    int thread_items[ItemsPerThread]{};

    block_load{}.load(device_input + items_in_previous_blocks, thread_items, valid_items, 0);

    // Example demonstrates summation therefore using rocprim::plus as the operator.
    int sum;
    block_reduce{}.reduce(thread_items, sum, rocprim::plus<int>());

    // Since all threads in the block have the same reduced value, only one thread per block
    // writes its value back to output in the global memory.
    if(threadIdx.x == 0)
    {
        device_output[blockIdx.x] = sum;
    }
}

int main()
{
    // Number of threads per block
    constexpr unsigned int block_size       = 256;
    constexpr unsigned int items_per_thread = 4;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;

    // Input problem size
    constexpr unsigned int size      = 10000;
    const unsigned int     grid_size = ceiling_div(size, items_per_block);

    // Allocate input vector on the host
    std::vector<int> h_in(size);
    // Initialize the input with consecutive numbers
    // Example: 0, 1, 2, 3, 4, 5, ..., size-1
    std::iota(h_in.begin(), h_in.end(), 0);

    std::cout << "Performing summation reduction of independent blocks using "
                 "rocprim::block_reduce on a vector of "
              << h_in.size() << " elements, where " << grid_size << " blocks take a maximum of "
              << items_per_block << " elements each" << std::endl;

    // Allocate the storage on the device that is needed to copy h_in to the device
    const size_t h_in_bytes = sizeof(int) * h_in.size();
    int*         d_in       = nullptr;
    HIP_CHECK(hipMalloc(&d_in, h_in_bytes));

    // Allocate storage for device output
    // The reduce sum produces a single scalar per block
    int* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int) * grid_size));

    // Initialize the device input data
    // Copy h_in from the host to device in d_in
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), h_in_bytes, hipMemcpyHostToDevice));

    // Running kernel
    reduce_sum_kernel<block_size, items_per_thread>
        <<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(d_in, d_out, size);

    // Allocate out vector on the host
    std::vector<int> h_out(grid_size);
    // Copy d_out from the device to host in h_out
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, sizeof(int) * grid_size, hipMemcpyDeviceToHost));

    std::cout << "Reduced Summation Result:\t" << format_range(h_out.begin(), h_out.end())
              << std::endl;
    // Compute results on the host CPU for verification
    std::vector<int> expected_result = reduce_sum_host(h_in, items_per_block, grid_size);
    std::cout << "Expected Summation Result:\t"
              << format_range(expected_result.begin(), expected_result.end()) << std::endl;

    // Free memory
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
}
