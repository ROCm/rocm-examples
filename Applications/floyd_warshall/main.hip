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

#include "cmdparser.hpp"
#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

/// \brief Implements the k-th (0 <= k < nodes) step of Floyd-Warshall algorithm. That is,
/// given a directed and weighted graph G = (V,E,w) (also complete in this example), it
/// computes the shortest path between every pair of vertices only considering as intermediate
/// nodes in the path the ones in the subset V' = {v_0,v_1,...,v_k} of V.
__global__ void floyd_warshall_kernel(unsigned int*      part_adjacency_matrix,
                                      unsigned int*      part_next_matrix,
                                      const unsigned int nodes,
                                      const unsigned int k)
{
    // Compute the vertices which shortest path each thread is going to process.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Get the current distance between the two vertices (only with intermediate nodes in
    // {v_0,v_1,...,v_{k-1}}) and compute the distance using node v_k as intermediate. Note that
    // d_x_k_y is the shortest path between x and y with node v_k as intermediate, because
    // otherwise we could find a shorter path between y and v_k or/and v_k and x using intermediate
    // nodes from {v_0,v_1,...,v_{k-1}} and thus contradicting the fact that the current paths
    // between those two pairs of nodes are already the shortest possible.
    int d_x_y   = part_adjacency_matrix[y * nodes + x];
    int d_x_k_y = part_adjacency_matrix[y * nodes + k] + part_adjacency_matrix[k * nodes + x];

    // If the path with intermediate nodes in {v_0, ..., v_{k-1}} is longer than the one
    // with intermediate node v_k, update matrices so the latter is selected as the
    // shortest path between x and y with intermediate nodes in {v_0, ..., v_k}.
    if(d_x_k_y < d_x_y)
    {
        part_adjacency_matrix[y * nodes + x] = d_x_k_y;
        part_next_matrix[y * nodes + x]      = k;
    }
}

/// \brief Reference CPU implementation of Floyd-Warshall algorithm for results verification.
void floyd_warshall_reference(unsigned int*      adjacency_matrix,
                              unsigned int*      next_matrix,
                              const unsigned int nodes)
{
    for(unsigned int k = 0; k < nodes; k++)
    {
        for(unsigned int x = 0; x < nodes; x++)
        {
            const unsigned int row_x = x * nodes;
            for(unsigned int y = 0; y < nodes; y++)
            {
                // d_x_y is the shortest distance from node x to node y with intermediate
                // nodes in {v_0, ..., v_{k-1}}. The other two are analogous.
                const unsigned int d_x_y = adjacency_matrix[row_x + y];
                const unsigned int d_x_k = adjacency_matrix[row_x + k];
                const unsigned int d_k_y = adjacency_matrix[k * nodes + y];

                // Shortest distance from node x to node y passing through node v_k.
                const unsigned int d_x_k_y = d_x_k + d_k_y;

                // If the path with intermediate nodes in {v_0, ..., v_{k-1}} is longer than the one
                // with intermediate node v_k, update matrices so the latter is selected as the
                // shortest path between x and y with intermediate nodes in {v_0, ..., v_k}.
                if(d_x_k_y < d_x_y)
                {
                    adjacency_matrix[row_x + y] = d_x_k_y;
                    next_matrix[row_x + y]      = k;
                }
            }
        }
    }
}

/// \brief Adds to a command line parser the necessary options for this example.
template<unsigned int BlockSize>
void configure_parser(cli::Parser& parser)
{
    // Default parameters.
    constexpr unsigned int nodes      = 16;
    constexpr unsigned int iterations = 1;

    static_assert(((nodes % BlockSize == 0)),
                  "Number of nodes must be a positive multiple of BlockSize");
    static_assert(((iterations > 0)), "Number of iterations must be at least 1");

    // Add options to the command line parser.
    parser.set_optional<unsigned int>("n", "nodes", nodes, "Number of nodes in the graph.");
    parser.set_optional<unsigned int>("i",
                                      "iterations",
                                      iterations,
                                      "Number of times the algorithm is executed.");
}

int main(int argc, char* argv[])
{
    // Number of threads in each kernel block dimension.
    constexpr unsigned int block_size = 16;

    // Parse user input.
    cli::Parser parser(argc, argv);
    configure_parser<block_size>(parser);
    parser.run_and_exit_if_error();

    // Get number of nodes and iterations from the command line, if provided.
    const unsigned int nodes      = parser.get<unsigned int>("n");
    const unsigned int iterations = parser.get<unsigned int>("i");

    // Check values provided.
    if(nodes % block_size)
    {
        std::cout << "Number of nodes must be a positive multiple of block_size ("
                  << std::to_string(block_size) << ")." << std::endl;
        return error_exit_code;
    }
    if(iterations == 0)
    {
        std::cout << "Number of iterations must be at least 1." << std::endl;
        return error_exit_code;
    }

    // Total number of elements and bytes of the input matrices.
    const unsigned int size       = nodes * nodes;
    const unsigned int size_bytes = nodes * nodes * sizeof(unsigned int);

    // Number of threads in each kernel block and number of blocks in the grid.
    const dim3 block_dim(block_size, block_size);
    const dim3 grid_dim(nodes / block_size, nodes / block_size);

    // Allocate host input adjacency matrix initialized with the increasing sequence 1,2,3,... .
    // Overwrite diagonal values (distance from a node to itself) to 0.
    std::vector<unsigned int> adjacency_matrix(size);
    std::iota(adjacency_matrix.begin(), adjacency_matrix.end(), 1);
    for(unsigned int x = 0; x < nodes; x++)
    {
        adjacency_matrix[x * nodes + x] = 0;
    }

    // Allocate host input matrix for the reconstruction of the paths obtained and initialize such
    // that the path from node x to node y is just the edge (x,y) for any pair of nodes x and y.
    std::vector<unsigned int> next_matrix(size);
    for(unsigned int x = 0; x < nodes; x++)
    {
        for(unsigned int y = 0; y < x; y++)
        {
            next_matrix[x * nodes + y] = x;
            next_matrix[y * nodes + x] = y;
        }
        next_matrix[x * nodes + x] = x;
    }

    // Allocate host memory for the CPU implementation and copy input data.
    std::vector<unsigned int> expected_adjacency_matrix(adjacency_matrix);
    std::vector<unsigned int> expected_next_matrix(next_matrix);

    // Declare host input (pinned) memory for incremental results from kernel executions.
    unsigned int* part_adjacency_matrix = nullptr;
    unsigned int* part_next_matrix      = nullptr;

    // Cumulative variable to compute the mean time per iteration of the algorithm.
    double kernel_time = 0;

    std::cout << "Executing Floyd-Warshall algorithm for " << iterations
              << " iterations with a complete graph of " << nodes << " nodes." << std::endl;

    // Allocate pinned host memory mapped to device memory.
    HIP_CHECK(hipHostMalloc(&part_adjacency_matrix, size_bytes, hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(&part_next_matrix, size_bytes, hipHostMallocMapped));

    // Copy memory to pinned memory region
    std::copy(adjacency_matrix.begin(), adjacency_matrix.end(), part_adjacency_matrix);
    std::copy(next_matrix.begin(), next_matrix.end(), part_next_matrix);

    // Allocate device memory
    unsigned int* d_adjacency_matrix;
    unsigned int* d_next_matrix;
    HIP_CHECK(hipMalloc(&d_adjacency_matrix, size_bytes));
    HIP_CHECK(hipMalloc(&d_next_matrix, size_bytes));

    // Create events to measure the execution time of the kernels.
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Run iterations times the Floyd-Warshall GPU algorithm.
    for(unsigned int i = 0; i < iterations; ++i)
    {
        // Copy input data from host to device memory.
        HIP_CHECK(hipMemcpy(d_adjacency_matrix,
                            part_adjacency_matrix,
                            size_bytes,
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_next_matrix, part_next_matrix, size_bytes, hipMemcpyHostToDevice));

        float kernel_ms{};

        // Floyd-Warshall GPU algorithm: launch Floyd-Warshall kernel for each node of the graph.
        for(unsigned int k = 0; k < nodes; ++k)
        {
            // Record the start event.
            HIP_CHECK(hipEventRecord(start, hipStreamDefault));

            // Launch Floyd-Warshall kernel on the default stream.
            floyd_warshall_kernel<<<grid_dim, block_dim, 0, hipStreamDefault>>>(d_adjacency_matrix,
                                                                                d_next_matrix,
                                                                                nodes,
                                                                                k);

            // Check if the kernel launch was successful.
            HIP_CHECK(hipGetLastError());

            // Record the stop event and wait until the kernel execution finishes.
            HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
            HIP_CHECK(hipEventSynchronize(stop));

            // Get the execution time of the kernel and add it to the total count.
            HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
            kernel_time += kernel_ms;
        }
    }
    // Free events used for time measurement
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Copy results back to host.
    HIP_CHECK(
        hipMemcpy(adjacency_matrix.data(), d_adjacency_matrix, size_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(next_matrix.data(), d_next_matrix, size_bytes, hipMemcpyDeviceToHost));

    // Free host memory.
    HIP_CHECK(hipHostFree(part_adjacency_matrix));
    HIP_CHECK(hipHostFree(part_next_matrix));

    // Free device memory
    HIP_CHECK(hipFree(d_adjacency_matrix));
    HIP_CHECK(hipFree(d_next_matrix));

    // Print the mean time per iteration (in miliseconds) of the algorithm.
    kernel_time /= iterations;
    std::cout << "The mean time needed for each iteration has been " << kernel_time << "ms."
              << std::endl;

    // Execute CPU algorithm.
    floyd_warshall_reference(expected_adjacency_matrix.data(), expected_next_matrix.data(), nodes);

    // Verify results.
    unsigned int errors = 0;
    std::cout << "Validating results with CPU implementation." << std::endl;
    for(unsigned int i = 0; i < size; ++i)
    {
        errors += (adjacency_matrix[i] - expected_adjacency_matrix[i] != 0);
        errors += (next_matrix[i] - expected_next_matrix[i] != 0);
    }

    if(errors)
    {
        std::cout << "Validation failed with " << errors << " errors." << std::endl;
        return error_exit_code;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }
}
