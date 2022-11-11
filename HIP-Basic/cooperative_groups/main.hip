// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#include <cstddef>
#include <cstdlib>

using namespace cooperative_groups;

/// \brief Summation of `unsigned int val`'s in `thread_group g` using shared memory `x`
__device__ unsigned int reduce_sum(thread_group g, unsigned int* x, unsigned int val)
{
    // Rank of this thread in the group
    const unsigned int group_thread_id = g.thread_rank();

    // We start with half the group size as active threads
    // Every iteration the number of active threads halves, until we processed all values
    for(unsigned int i = g.size() / 2; i > 0; i /= 2)
    {
        // Store value for this thread in a shared, temporary array
        x[group_thread_id] = val;

        // Synchronize all threads in the group
        g.sync();

        // If our thread is still active, sum with its counterpart in the other half
        if(group_thread_id < i)
        {
            val += x[group_thread_id + i];
        }

        // Synchronize all threads in the group
        g.sync();
    }

    // Only the first thread returns a valid value
    if(g.thread_rank() == 0)
        return val;
    else
        return 0;
}

/// \brief A vector reduction kernel showcasing the use of cooperative groups.
/// - First we showcase the use of threadBlockGroup.
/// - Second we showcase the use of `tiled_partition<>()`.
/// \param partition_size The number of elements in a cooperative group's tiled_partition.
template<unsigned int PartitionSize>
__global__ void vector_reduce_kernel(const unsigned int* d_vector,
                                     unsigned int*       d_block_reduced_vector,
                                     unsigned int*       d_partition_reduced_vector)
{
    // threadBlockGroup consists of all threads in the block
    thread_block thread_block_group = this_thread_block();

    // Workspace array in shared memory required for reduction
    __shared__ unsigned int workspace[2048];

    unsigned int output;

    // Input to reduce
    const unsigned int input = d_vector[thread_block_group.thread_rank()];

    // Perform reduction
    output = reduce_sum(thread_block_group, workspace, input);

    // Only the first thread returns a valid value
    if(thread_block_group.thread_rank() == 0)
    {
        d_block_reduced_vector[0] = output;
    }

    // Every custom_partition group consists of 16 threads
    thread_block_tile<PartitionSize> custom_partition
        = tiled_partition<PartitionSize>(thread_block_group);

    // To make sure every partition has its own piece of shared memory it can work with
    const unsigned int group_offset
        = thread_block_group.thread_rank() - custom_partition.thread_rank();

    // Perform reduction
    output = reduce_sum(custom_partition, &workspace[group_offset], input);

    // Only the first thread in each partition returns a valid value
    if(custom_partition.thread_rank() == 0)
    {
        const unsigned int partition_id          = thread_block_group.thread_rank() / PartitionSize;
        d_partition_reduced_vector[partition_id] = output;
    }
    return;
}

// Host side function to perform the same reductions as executed on the GPU
std::vector<unsigned int> ref_reduced(const unsigned int        partition_size,
                                      std::vector<unsigned int> input)
{
    const unsigned int        input_size  = input.size();
    const unsigned int        result_size = input_size / partition_size;
    std::vector<unsigned int> result(result_size);

    for(unsigned int i = 0; i < result_size; i++)
    {
        unsigned int partition_result = 0;
        for(unsigned int j = 0; j < partition_size; j++)
        {
            partition_result += input[partition_size * i + j];
        }
        result[i] = partition_result;
    }

    return result;
}

int main()
{
#ifdef __HIP_PLATFORM_AMD__
    int device               = 0;
    int supports_coop_launch = 0;
    // Check support
    // Use hipDeviceAttributeCooperativeMultiDeviceLaunch when launching across multiple devices
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(
        hipDeviceGetAttribute(&supports_coop_launch, hipDeviceAttributeCooperativeLaunch, device));
    if(!supports_coop_launch)
    {
        std::cout << "Skipping, device " << device << " does not support cooperative groups"
                  << std::endl;
        return 0;
    }
#endif

    // Number of blocks to launch.
    constexpr unsigned int num_blocks = 1;

    // Number of threads in each kernel block.
    constexpr unsigned int threads_per_block = 64;

    // Total element count of the input vector.
    constexpr unsigned int size = num_blocks * threads_per_block;

    // Total elements count of a tiled_partition.
    constexpr unsigned int partition_size = 16;

    // Total size (in bytes) of the input vector.
    constexpr size_t size_bytes = sizeof(unsigned int) * size;

    static_assert(threads_per_block % partition_size == 0,
                  "threads_per_block must be a multiple of partition_size");

    // Allocate host vectors.
    std::vector<unsigned int> h_vector(size);
    std::vector<unsigned int> h_block_reduced(num_blocks);
    std::vector<unsigned int> h_partition_reduced(threads_per_block / partition_size);

    // Set up input data.
    for(unsigned int i = 0; i < size; i++)
    {
        h_vector[i] = i;
    }

    // Allocate device memory for the input and output matrices.
    unsigned int* d_vector{};
    unsigned int* d_block_reduced{};
    unsigned int* d_partition_reduced{};
    HIP_CHECK(hipMalloc(&d_vector, size_bytes));
    HIP_CHECK(hipMalloc(&d_block_reduced, sizeof(unsigned int) * h_block_reduced.size()));
    HIP_CHECK(hipMalloc(&d_partition_reduced, sizeof(unsigned int) * h_partition_reduced.size()));

    // Transfer the input vector to the device memory.
    HIP_CHECK(hipMemcpy(d_vector, h_vector.data(), size_bytes, hipMemcpyHostToDevice));

    void* params[] = {&d_vector, &d_block_reduced, &d_partition_reduced};
    // Launching kernel from host.
    HIP_CHECK(hipLaunchCooperativeKernel(vector_reduce_kernel<partition_size>,
                                         dim3(num_blocks),
                                         dim3(threads_per_block),
                                         params,
                                         0,
                                         hipStreamDefault));

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Transfer the result back to the host.
    HIP_CHECK(hipMemcpy(h_block_reduced.data(),
                        d_block_reduced,
                        sizeof(unsigned int) * h_block_reduced.size(),
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipMemcpy(h_partition_reduced.data(),
                        d_partition_reduced,
                        sizeof(unsigned int) * h_partition_reduced.size(),
                        hipMemcpyDeviceToHost));

    // Free the resources on the device.
    HIP_CHECK(hipFree(d_vector));
    HIP_CHECK(hipFree(d_block_reduced));
    HIP_CHECK(hipFree(d_partition_reduced));

    // Perform the reference (CPU) calculation.
    std::vector<unsigned int> ref_block_reduced     = ref_reduced(threads_per_block, h_vector);
    std::vector<unsigned int> ref_partition_reduced = ref_reduced(partition_size, h_vector);

    // Check the results' validity.
    unsigned int errors{};
    for(unsigned int i = 0; i < h_block_reduced.size(); i++)
    {
        errors += (h_block_reduced[i] != ref_block_reduced[i]);
    }
    for(unsigned int i = 0; i < h_partition_reduced.size(); i++)
    {
        errors += (h_partition_reduced[i] != ref_partition_reduced[i]);
    }

    if(errors)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        return error_exit_code;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }
}