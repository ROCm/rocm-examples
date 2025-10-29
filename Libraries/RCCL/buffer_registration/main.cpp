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
    parser.set_optional<size_t>("s", "size", 1024 * 1024, "Total number of elements");
    parser.set_optional<int>("l", "layers", 4, "Number of layers to register");
    parser.set_optional<int>("i", "iterations", 5, "Number of iterations");
}

int main(int argc, const char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    int          num_ranks  = parser.get<int>("n");
    const size_t size       = parser.get<size_t>("s");
    const int    num_layers = parser.get<int>("l");
    const int    iterations = parser.get<int>("i");

    // Validate arguments
    if(size <= 0)
    {
        std::cerr << "Error: size must be positive" << std::endl;
        return error_exit_code;
    }

    if(num_layers <= 0)
    {
        std::cerr << "Error: layers must be positive" << std::endl;
        return error_exit_code;
    }

    if(iterations <= 0)
    {
        std::cerr << "Error: iterations must be positive" << std::endl;
        return error_exit_code;
    }

    // Calculate layer size
    const size_t layer_size = size / num_layers;
    if(layer_size * num_layers != size)
    {
        std::cerr << "Error: size must be divisible by number of layers" << std::endl;
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

    std::cout << "RCCL Buffer Registration Example" << std::endl;
    std::cout << "Number of ranks: " << num_ranks << std::endl;
    std::cout << "Number of layers: " << num_layers << std::endl;
    std::cout << "Elements per layer: " << (layer_size / 1024) << "K" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

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
    // STEP 3: Allocate Memory for All Layers (ncclMemAlloc)
    // ========================================================================

    // Allocate buffers for each layer and rank
    std::vector<std::vector<float*>> device_inputs(num_ranks);
    std::vector<std::vector<float*>> device_outputs(num_ranks);
    std::vector<std::vector<void*>>  input_handles(num_ranks);
    std::vector<std::vector<void*>>  output_handles(num_ranks);

    for(int rank = 0; rank < num_ranks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        device_inputs[rank].resize(num_layers);
        device_outputs[rank].resize(num_layers);
        input_handles[rank].resize(num_layers);
        output_handles[rank].resize(num_layers);

        for(int layer = 0; layer < num_layers; ++layer)
        {
            RCCL_CHECK(
                ncclMemAlloc((void**)&device_inputs[rank][layer], layer_size * sizeof(float)));
            RCCL_CHECK(
                ncclMemAlloc((void**)&device_outputs[rank][layer], layer_size * sizeof(float)));
        }
    }

    std::cout << "Allocated " << num_layers << " layers per rank using RCCL memory management"
              << std::endl;
    std::cout << "Layer size: " << (layer_size * sizeof(float) / 1024.0 / 1024.0) << " MB"
              << std::endl;
    std::cout << "Total memory per rank: " << (size * sizeof(float) / 1024.0 / 1024.0) << " MB"
              << std::endl;

    // ========================================================================
    // STEP 4: Register All Buffers with Communicators
    // ========================================================================

    // Buffer registration optimizes repeated operations on the same buffers
    // Useful for training loops that reuse gradient buffers across iterations
    // Register buffers once before the loop - This eliminates per-operation registration overhead
    std::cout << "\nRegistering buffers with communicators..." << std::endl;

    for(int rank = 0; rank < num_ranks; ++rank)
    {
        for(int layer = 0; layer < num_layers; ++layer)
        {
            RCCL_CHECK(ncclCommRegister(comms[rank],
                                        device_inputs[rank][layer],
                                        layer_size * sizeof(float),
                                        &input_handles[rank][layer]));
            RCCL_CHECK(ncclCommRegister(comms[rank],
                                        device_outputs[rank][layer],
                                        layer_size * sizeof(float),
                                        &output_handles[rank][layer]));
        }
        std::cout << "  Rank " << rank << " registered " << (num_layers * 2) << " buffers"
                  << std::endl;
    }

    std::cout << "All buffers registered successfully" << std::endl;

    // ========================================================================
    // STEP 5: Initialize Data
    // ========================================================================

    // Initialize each layer with rank-specific pattern
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        for(int layer = 0; layer < num_layers; ++layer)
        {
            std::vector<float> host_data(layer_size);

            for(size_t i = 0; i < layer_size; ++i)
            {
                host_data[i] = static_cast<float>(rank * 100 + layer * 10 + (i % 10));
            }

            HIP_CHECK(hipMemcpy(device_inputs[rank][layer],
                                host_data.data(),
                                layer_size * sizeof(float),
                                hipMemcpyHostToDevice));
        }
    }

    // Print sample of initial data
    std::cout << "\nInitial data sample (first 5 elements of first layer):" << std::endl;
    for(int rank = 0; rank < num_ranks; ++rank)
    {
        std::vector<float> sample(5);
        HIP_CHECK(hipMemcpy(sample.data(),
                            device_inputs[rank][0],
                            5 * sizeof(float),
                            hipMemcpyDeviceToHost));
        std::cout << "  Rank " << rank << " layer[0]: [";
        for(int i = 0; i < 5; ++i)
        {
            std::cout << sample[i];
            if(i < 4)
            {
                std::cout << ", ";
            }
        }
        std::cout << ", ...]" << std::endl;
    }

    // ========================================================================
    // STEP 6: Perform Repeated Operations on Registered Buffers
    // ========================================================================

    std::cout << "\nPerforming " << iterations << " iterations with registered buffers..."
              << std::endl;

    for(int iter = 0; iter < iterations; ++iter)
    {
        std::cout << "Iteration " << (iter + 1) << "/" << iterations << std::endl;

        // Perform allreduce on each layer
        RCCL_CHECK(ncclGroupStart());
        for(int rank = 0; rank < num_ranks; ++rank)
        {
            HIP_CHECK(hipSetDevice(rank));
            for(int layer = 0; layer < num_layers; ++layer)
            {
                RCCL_CHECK(ncclAllReduce(device_inputs[rank][layer],
                                         device_outputs[rank][layer],
                                         layer_size,
                                         ncclFloat,
                                         ncclSum,
                                         comms[rank],
                                         streams[rank]));
            }
        }
        RCCL_CHECK(ncclGroupEnd());

        // Synchronize
        for(int rank = 0; rank < num_ranks; ++rank)
        {
            HIP_CHECK(hipStreamSynchronize(streams[rank]));
        }
    }

    std::cout << "All iterations completed" << std::endl;

    // ========================================================================
    // STEP 7: Verify Results
    // ========================================================================

    std::cout << "\nVerification:" << std::endl;
    bool all_passed = true;

    // Verify each layer
    for(int layer = 0; layer < num_layers; ++layer)
    {
        for(int rank = 0; rank < num_ranks; ++rank)
        {
            std::vector<float> host_output(layer_size);
            HIP_CHECK(hipMemcpy(host_output.data(),
                                device_outputs[rank][layer],
                                layer_size * sizeof(float),
                                hipMemcpyDeviceToHost));

            // Expected: sum across all ranks
            bool layer_passed = true;
            for(size_t i = 0; i < layer_size && layer_passed; ++i)
            {
                float expected = 0.0f;
                for(int r = 0; r < num_ranks; ++r)
                {
                    expected += static_cast<float>(r * 100 + layer * 10 + (i % 10));
                }

                if(std::abs(host_output[i] - expected) > 1e-5)
                {
                    layer_passed = false;
                    all_passed   = false;
                }
            }

            if(rank == 0)
            {
                std::cout << "Layer " << (layer + 1) << ": " << (layer_passed ? "PASSED" : "FAILED")
                          << std::endl;
            }
        }
    }

    if(all_passed)
    {
        std::cout << "\n=== Buffer Registration verification PASSED ===" << std::endl;
    }
    else
    {
        std::cout << "\n=== Buffer Registration verification FAILED ===" << std::endl;
    }

    // ========================================================================
    // STEP 8: Deregister Buffers
    // ========================================================================

    // Deregister before freeing memory or destroying communicators
    // Important for proper resource management
    std::cout << "\nDeregistering buffers..." << std::endl;

    for(int rank = 0; rank < num_ranks; ++rank)
    {
        for(int layer = 0; layer < num_layers; ++layer)
        {
            RCCL_CHECK(ncclCommDeregister(comms[rank], input_handles[rank][layer]));
            RCCL_CHECK(ncclCommDeregister(comms[rank], output_handles[rank][layer]));
        }
        std::cout << "  Rank " << rank << " deregistered " << (num_layers * 2) << " buffers"
                  << std::endl;
    }

    std::cout << "All buffers deregistered successfully" << std::endl;

    // ========================================================================
    // STEP 9: Cleanup Resources
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

        for(int layer = 0; layer < num_layers; ++layer)
        {
            RCCL_CHECK(ncclMemFree(device_inputs[rank][layer]));
            RCCL_CHECK(ncclMemFree(device_outputs[rank][layer]));
        }

        RCCL_CHECK(ncclCommFinalize(comms[rank]));
        RCCL_CHECK(ncclCommDestroy(comms[rank]));
    }

    std::cout << "\n=== Buffer Registration example completed successfully ===" << std::endl;

    return all_passed ? 0 : error_exit_code;
}
