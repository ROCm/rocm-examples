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

#include <iostream>
#include <vector>

void configure_parser(cli::Parser& parser)
{
    parser.set_optional<int>("n", "num-ranks", 0, "Number of ranks (0 = use all available GPUs)");
    parser.set_optional<size_t>("s", "size", 1024, "Number of elements for communication");
}

int main(int argc, const char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    int          num_ranks = parser.get<int>("n");
    const size_t size      = parser.get<size_t>("s");

    // Validate size argument
    if(size <= 0)
    {
        std::cerr << "Error: size must be positive" << std::endl;
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

    std::cout << "RCCL AllReduce Example" << std::endl;
    std::cout << "Running with " << num_ranks << " rank(s)" << std::endl;
    std::cout << "Array size: " << size << " elements (" << (size * sizeof(float)) << " bytes)"
              << std::endl;

    if(num_ranks == 1)
    {
        std::cout << "\nNote: Running with single rank. AllReduce will be a no-op (input = output)."
                  << std::endl;
        std::cout << "For meaningful collective operations, run on a system with multiple GPUs.\n"
                  << std::endl;
    }

    // ========================================================================
    // STEP 1: Initialize RCCL Communicators and Detect GPUs
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
    // STEP 2: Create HIP Streams for Each Rank
    // ========================================================================

    // Create HIP streams for each rank
    std::vector<hipStream_t> streams(num_ranks);
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        HIP_CHECK(hipStreamCreate(&streams[rank]));
    }

    // ========================================================================
    // STEP 3: Allocate Memory Using RCCL Memory Management
    // ========================================================================

    // Allocate device and host memory for each rank
    std::vector<float*>             device_inputs(num_ranks);
    std::vector<float*>             device_outputs(num_ranks);
    std::vector<std::vector<float>> host_inputs(num_ranks);
    std::vector<std::vector<float>> host_outputs(num_ranks);

    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));

        // Allocate device memory using RCCL memory management
        RCCL_CHECK(ncclMemAlloc((void**)&device_inputs[rank], size * sizeof(float)));
        RCCL_CHECK(ncclMemAlloc((void**)&device_outputs[rank], size * sizeof(float)));

        // Allocate and initialize host memory
        host_inputs[rank].resize(size);
        host_outputs[rank].resize(size);

        // Each rank has unique input: rank value at each element
        for(size_t i = 0; i < size; ++i)
        {
            host_inputs[rank][i] = static_cast<float>(rank + 1);
        }

        // Copy input to device
        HIP_CHECK(hipMemcpy(device_inputs[rank],
                            host_inputs[rank].data(),
                            size * sizeof(float),
                            hipMemcpyHostToDevice));

        // Print initial data for first few elements
        if(rank == 0)
        {
            std::cout << "\nInitial data (first 10 elements):" << std::endl;
        }
        std::cout << "  Rank " << rank << " input: [";
        const size_t print_count = std::min(size, static_cast<size_t>(10));
        for(size_t i = 0; i < print_count; ++i)
        {
            std::cout << host_inputs[rank][i];
            if(i < print_count - 1)
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
    // STEP 4: Initialize Data and Copy to Device
    // ========================================================================
    std::cout << "Buffer size: " << (size * sizeof(float) / 1024.0 / 1024.0) << " MB per rank"
              << std::endl;
    std::cout << "\nPerforming AllReduce (sum) operation..." << std::endl;

    // ========================================================================
    // STEP 5: Perform AllReduce Operation Using Group API
    // ========================================================================

    // Perform allreduce operations for all ranks
    // Use ncclGroupStart/ncclGroupEnd to launch all operations together
    RCCL_CHECK(ncclGroupStart());
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        RCCL_CHECK(ncclAllReduce(device_inputs[rank],
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
    // STEP 6: Synchronize and Verify Results
    // ========================================================================

    // Copy results back to host and verify
    bool all_passed = true;

    // Calculate expected result: sum of all rank inputs
    // Each rank contributes (rank+1) at each element
    // So expected = sum(1, 2, ..., num_ranks) = num_ranks * (num_ranks + 1) / 2
    float expected = 0.0f;
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        expected += static_cast<float>(rank + 1);
    }

    std::cout << "\nResults (first 10 elements):" << std::endl;
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipMemcpy(host_outputs[rank].data(),
                            device_outputs[rank],
                            size * sizeof(float),
                            hipMemcpyDeviceToHost));

        // Print output
        std::cout << "  Rank " << rank << " output: [";
        const size_t print_count = std::min(size, static_cast<size_t>(10));
        for(size_t i = 0; i < print_count; ++i)
        {
            std::cout << host_outputs[rank][i];
            if(i < print_count - 1)
            {
                std::cout << ", ";
            }
        }
        if(size > 10)
        {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;

        // Verify results (all ranks should have identical results after allreduce)
        bool rank_passed = true;
        for(size_t i = 0; i < size; ++i)
        {
            if(std::abs(host_outputs[rank][i] - expected) > 1e-5)
            {
                std::cerr << "Rank " << rank << " verification FAILED at index " << i
                          << ": expected " << expected << ", got " << host_outputs[rank][i]
                          << std::endl;
                rank_passed = false;
                all_passed  = false;
                break;
            }
        }

        if(rank_passed && rank == 0)
        {
            std::cout << "\nExpected sum: " << expected << " (sum of ranks 1.." << num_ranks << ")"
                      << std::endl;
        }
    }

    if(all_passed)
    {
        std::cout << "\n=== AllReduce verification PASSED for all ranks! ===" << std::endl;
    }
    else
    {
        std::cout << "\n=== AllReduce verification FAILED for one or more ranks! ===" << std::endl;
    }

    // ========================================================================
    // STEP 7: Cleanup Resources in Proper Order
    // ========================================================================
    // Cleanup order: destroy streams → free memory → finalize comm → destroy comm
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        HIP_CHECK(hipStreamDestroy(streams[rank]));
        RCCL_CHECK(ncclMemFree(device_inputs[rank]));
        RCCL_CHECK(ncclMemFree(device_outputs[rank]));
        RCCL_CHECK(ncclCommFinalize(comms[rank]));
        RCCL_CHECK(ncclCommDestroy(comms[rank]));
    }

    return all_passed ? 0 : error_exit_code;
}
