// MIT License
//
// Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iomanip>
#include <iostream>
#include <vector>

/// \brief Performs pair-wise multiplication of two vectors and stores the result into
/// a third vector
__global__ void
    pairwise_product_kernel(float* C, const float* A, const float* B, const unsigned int size)
{
    const unsigned int global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(global_thread_idx < size)
    {
        C[global_thread_idx] = A[global_thread_idx] * B[global_thread_idx];
    }
}

/// \brief Host function to calculate and print occupancy based on a given
/// block size and kernel function
void print_occupancy(const int block_size)
{
    hipDeviceProp_t dev_prop;
    HIP_CHECK(hipGetDeviceProperties(&dev_prop, 0));

    if(dev_prop.maxThreadsPerMultiProcessor)
    {
        // Calculate Occupancy
        int num_blocks = 0;
        HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks,
                                                               pairwise_product_kernel,
                                                               block_size,
                                                               0));
        std::cout << "Theoretical Occupancy is "
                  << double_precision(static_cast<double>(num_blocks) * block_size
                                          / dev_prop.maxThreadsPerMultiProcessor * 100,
                                      3)
                  << "%" << std::endl;
    }
}

void deploy_kernel_manual_parameters(float*             d_C,
                                     const float*       d_A,
                                     const float*       d_B,
                                     const unsigned int size)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Manually set the number of threads in a block
    constexpr unsigned int block_size = 32;

    // Round up the blocks in case size is not evenly divisible
    const unsigned int grid_size = (size + (block_size - 1)) / block_size;

    std::cout << "\nManual Configuration with Block Size " << block_size << '\n';

    // Record the start event
    HIP_CHECK(hipEventRecord(start, nullptr));

    // Launch the kernel
    pairwise_product_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(d_C,
                                                                                        d_A,
                                                                                        d_B,
                                                                                        size);

    // Record the stop event
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));

    float event_ms;
    HIP_CHECK(hipEventElapsedTime(&event_ms, start, stop));
    std::cout << "Kernel Execution Time: " << double_precision(event_ms, 2) << " ms" << std::endl;

    // Destroy the events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    print_occupancy(block_size);
}

void deploy_kernel_automatic_parameters(float*             d_C,
                                        const float*       d_A,
                                        const float*       d_B,
                                        const unsigned int size)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    int min_grid_size = 0;
    int block_size    = 0;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&min_grid_size,
                                                &block_size,
                                                pairwise_product_kernel,
                                                0,
                                                0));
    std::cout << "\nAutomatic Configuation based on hipOccupancyMaxPotentialBlockSize \n";
    std::cout << "Suggested Block Size is " << block_size << ", Minimum Grid Size is "
              << min_grid_size << std::endl;

    // Round up the blocks in case size is not evenly divisible
    const unsigned int grid_size = (size + (block_size - 1)) / block_size;

    // Record the start event
    HIP_CHECK(hipEventRecord(start, nullptr));

    // Launch the kernel
    pairwise_product_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(d_C,
                                                                                        d_A,
                                                                                        d_B,
                                                                                        size);

    // Record the stop event
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));

    float event_ms;
    HIP_CHECK(hipEventElapsedTime(&event_ms, start, stop));
    std::cout << "Kernel Execution Time: " << double_precision(event_ms, 2) << " ms" << std::endl;

    // Destroy the events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    print_occupancy(block_size);
}

int main()
{
    constexpr int size = 10000;

    std::vector<float> h_A(size);
    std::vector<float> h_B(size);

    // Initialize the input data
    for(int i = 0; i < size; i++)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    // Allocate the memory on the device side
    const size_t size_in_bytes = sizeof(float) * h_A.size();
    HIP_CHECK(hipMalloc(&d_A, size_in_bytes));
    HIP_CHECK(hipMalloc(&d_B, size_in_bytes));
    HIP_CHECK(hipMalloc(&d_C, size_in_bytes));

    // Memory transfer from host to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_in_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_in_bytes, hipMemcpyHostToDevice));

    // Kernel launch with manual/default block size
    deploy_kernel_manual_parameters(d_C, d_A, d_B, size);

    // Kernel launch with the block size suggested by hipOccupancyMaxPotentialblock_size
    deploy_kernel_automatic_parameters(d_C, d_A, d_B, size);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}
