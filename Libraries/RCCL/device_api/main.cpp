// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"
#include "rccl_utils.hpp"

#include <rccl/rccl.h>

#include <algorithm>
#include <cmath>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Kernel configuration for gradient clipping kernel
#define CTA_COUNT 16
#define THREADS_PER_CTA 512

// Gradient clipping kernel - demonstrates computation that would benefit from fusion
// In a real Device API scenario, this would be fused with collective communication
__global__ void
    gradient_clip_kernel(float* input, float* output, size_t count, float clip_threshold)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Grid stride loop over all elements
    for(size_t i = tid; i < count; i += stride)
    {
        float val = input[i];
        // Clip gradient to [-threshold, +threshold]
        output[i] = fmaxf(-clip_threshold, fminf(clip_threshold, val));
    }
}

void configure_parser(cli::Parser& parser)
{
    parser.set_optional<int>("n", "num-ranks", 0, "Number of ranks (0 = use all available GPUs)");
    parser.set_optional<size_t>("s", "size", 1024 * 1024, "Number of elements");
    parser.set_optional<float>("t", "threshold", 1.0f, "Gradient clipping threshold");
}

int main(int argc, const char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    int          num_ranks      = parser.get<int>("n");
    const size_t size           = parser.get<size_t>("s");
    const float  clip_threshold = parser.get<float>("t");

    // Validate arguments
    if(size <= 0)
    {
        std::cerr << "Error: size must be positive" << std::endl;
        return error_exit_code;
    }

    if(clip_threshold <= 0.0f)
    {
        std::cerr << "Error: threshold must be positive" << std::endl;
        return error_exit_code;
    }

    // Detect available GPUs
    const int num_gpus = detect_num_gpus();
    std::cout << "Available GPUs: " << num_gpus << std::endl;

    if(num_gpus == 0)
    {
        std::cerr << "Error: No GPUs detected" << std::endl;
        return error_exit_code;
    }

    // Set num_ranks to num_gpus if not specified or if too large
    if(num_ranks <= 0)
    {
        num_ranks = num_gpus;
        std::cout << "Using all " << num_ranks << " available GPU(s)" << std::endl;
    }
    else if(num_ranks > num_gpus)
    {
        std::cout << "Warning: Requested " << num_ranks << " ranks but only " << num_gpus
                  << " GPU(s) available." << std::endl;
        std::cout << "Setting num_ranks to " << num_gpus << std::endl;
        num_ranks = num_gpus;
    }

    std::cout << "RCCL Gradient Clipping + AllReduce Example" << std::endl;
    std::cout << "(Demonstrating concepts for Device API fusion)" << std::endl;
    std::cout << "Number of ranks: " << num_ranks << std::endl;
    std::cout << "Array size: " << (size / 1024 / 1024) << "M elements" << std::endl;
    std::cout << "Clipping threshold: " << clip_threshold << std::endl;

    if(num_ranks == 1)
    {
        std::cout << "\nNote: Running with single rank. AllReduce will be a no-op (input = output)."
                  << std::endl;
        std::cout << "For meaningful collective operations, run on a system with multiple GPUs.\n"
                  << std::endl;
    }

    // ========================================================================
    // STEP 1: Initialize RCCL Communicators
    // ========================================================================

    // Initialize RCCL communicators (one per GPU)
    std::vector<ncclComm_t> comms(num_ranks);
    RCCL_CHECK(ncclCommInitAll(comms.data(), num_ranks, nullptr));
    std::cout << "Initialized " << num_ranks << " RCCL communicator(s)" << std::endl;

    // Print communicator information using query APIs
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        int user_rank, num_ranks_query, device_id;
        RCCL_CHECK(ncclCommUserRank(comms[rank], &user_rank));
        RCCL_CHECK(ncclCommCount(comms[rank], &num_ranks_query));
        RCCL_CHECK(ncclCommCuDevice(comms[rank], &device_id));
        std::cout << "  Rank " << rank << ": user_rank=" << user_rank
                  << ", num_ranks=" << num_ranks_query << ", device=" << device_id << std::endl;
    }

    // ========================================================================
    // STEP 2: Create HIP Streams
    // ========================================================================

    // Create HIP streams for each rank
    std::vector<hipStream_t> streams(num_ranks);
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        HIP_CHECK(hipStreamCreate(&streams[rank]));
    }

    // ========================================================================
    // STEP 3: Allocate Memory (ncclMemAlloc)
    // ========================================================================

    // Allocate device memory using RCCL memory management
    // This demonstrates the use of RCCL's memory allocation APIs
    std::vector<float*> device_inputs(num_ranks);
    std::vector<float*> device_clipped(num_ranks);
    std::vector<float*> device_outputs(num_ranks);

    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        RCCL_CHECK(ncclMemAlloc((void**)&device_inputs[rank], size * sizeof(float)));
        RCCL_CHECK(ncclMemAlloc((void**)&device_clipped[rank], size * sizeof(float)));
        RCCL_CHECK(ncclMemAlloc((void**)&device_outputs[rank], size * sizeof(float)));
    }

    std::cout << "Allocated RCCL-managed memory: " << (size * sizeof(float) * 3 / 1024.0 / 1024.0)
              << " MB per rank" << std::endl;

    // ========================================================================
    // STEP 4: Initialize Data with Gradient-like Patterns
    // ========================================================================

    std::cout << "\nInitializing data with gradient-like patterns..." << std::endl;
    std::cout << "Clipping threshold: " << clip_threshold << std::endl;

    for(int rank = 0; rank < num_ranks; ++rank)
    {
        std::vector<float> host_data(size);

        // Initialize with values that will be clipped
        for(size_t i = 0; i < size; ++i)
        {
            // Pattern: some values exceed threshold to demonstrate clipping
            float val    = static_cast<float>(rank * 2.0f + (i % 5) - 2.0f);
            host_data[i] = val;
        }

        HIP_CHECK(hipMemcpy(device_inputs[rank],
                            host_data.data(),
                            size * sizeof(float),
                            hipMemcpyHostToDevice));

        // Print sample
        std::cout << "  Rank " << rank << " input (first 10): [";
        for(size_t i = 0; i < std::min(size, static_cast<size_t>(10)); ++i)
        {
            std::cout << host_data[i];
            if(i < 9 && i < size - 1)
            {
                std::cout << ", ";
            }
        }
        if(size > 10)
        {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;
    }

    // ========================================================================
    // STEP 5: Launch Gradient Clipping Kernel
    // ========================================================================

    std::cout << "\nLaunching gradient clipping kernel..." << std::endl;
    std::cout << "Kernel config: " << CTA_COUNT << " CTAs, " << THREADS_PER_CTA
              << " threads per CTA" << std::endl;

    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));

        gradient_clip_kernel<<<CTA_COUNT, THREADS_PER_CTA, 0, streams[rank]>>>(device_inputs[rank],
                                                                               device_clipped[rank],
                                                                               size,
                                                                               clip_threshold);

        HIP_CHECK(hipGetLastError());
    }

    // Synchronize all streams
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipStreamSynchronize(streams[rank]));
    }

    std::cout << "Gradient clipping completed" << std::endl;

    // ========================================================================
    // STEP 6: Perform AllReduce on Clipped Gradients
    // ========================================================================

    // In a real Device API scenario, steps 5 and 6 would be fused into a single kernel
    // This demonstrates the separation that Device API aims to eliminate
    std::cout << "\nPerforming AllReduce on clipped gradients..." << std::endl;
    std::cout << "Note: In Device API, clipping and AllReduce would be fused in one kernel"
              << std::endl;

    // Group all allreduce operations for better performance
    RCCL_CHECK(ncclGroupStart());
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        RCCL_CHECK(ncclAllReduce(device_clipped[rank],
                                 device_outputs[rank],
                                 size,
                                 ncclFloat,
                                 ncclSum,
                                 comms[rank],
                                 streams[rank]));
    }
    RCCL_CHECK(ncclGroupEnd());

    // Synchronize all streams
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipStreamSynchronize(streams[rank]));
    }

    std::cout << "AllReduce operation completed" << std::endl;

    // ========================================================================
    // STEP 7: Verify Results
    // ========================================================================

    std::cout << "\nVerification:" << std::endl;

    bool all_passed = true;
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        std::vector<float> host_output(size);
        HIP_CHECK(hipMemcpy(host_output.data(),
                            device_outputs[rank],
                            size * sizeof(float),
                            hipMemcpyDeviceToHost));

        // Calculate expected: sum of clipped values from all ranks
        bool rank_passed = true;
        for(size_t i = 0; i < size && rank_passed; ++i)
        {
            float expected = 0.0f;
            for(int r = 0; r < num_ranks; ++r)
            {
                float val = static_cast<float>(r * 2.0f + (i % 5) - 2.0f);
                // Apply clipping
                val = std::max(-clip_threshold, std::min(clip_threshold, val));
                expected += val;
            }

            if(std::abs(host_output[i] - expected) > 1e-5)
            {
                rank_passed = false;
                all_passed  = false;
            }
        }

        std::cout << "  Rank " << rank << ": " << (rank_passed ? "PASSED" : "FAILED") << std::endl;

        if(rank == 0)
        {
            std::cout << "    Sample output: [";
            for(size_t i = 0; i < std::min(size, static_cast<size_t>(10)); ++i)
            {
                std::cout << host_output[i];
                if(i < 9 && i < size - 1)
                {
                    std::cout << ", ";
                }
            }
            if(size > 10)
            {
                std::cout << ", ...";
            }
            std::cout << "]" << std::endl;
        }
    }

    // ========================================================================
    // STEP 8: Cleanup Resources
    // ========================================================================

    // Cleanup order:
    // 1. Free device memory (ncclMemFree)
    // 2. Destroy streams
    // 3. Finalize communicators (ncclCommFinalize)
    // 4. Destroy communicators (ncclCommDestroy)
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        HIP_CHECK(hipStreamDestroy(streams[rank]));
        RCCL_CHECK(ncclMemFree(device_inputs[rank]));
        RCCL_CHECK(ncclMemFree(device_clipped[rank]));
        RCCL_CHECK(ncclMemFree(device_outputs[rank]));
        RCCL_CHECK(ncclCommFinalize(comms[rank]));
        RCCL_CHECK(ncclCommDestroy(comms[rank]));
    }

    std::cout << "\n=== Gradient Clipping + AllReduce example completed ";
    if(all_passed)
    {
        std::cout << "successfully ===" << std::endl;
    }
    else
    {
        std::cout << "with errors ===" << std::endl;
    }

    return all_passed ? 0 : error_exit_code;
}
