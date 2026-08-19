// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/cooperative_groups/hip_scan.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#include <cstddef>
#include <cstdlib>

namespace cg = cooperative_groups;

/// \brief Computes an inclusive and an exclusive prefix sum (scan) over the input array.
///
/// The block is partitioned into warp-sized `thread_block_tile`s, and each tile performs
/// an independent scan over its own `TileSize` contiguous elements using the cooperative
/// groups scan API. Instantiating the tile at the device warp size lets the scan use the
/// DPP-accelerated hardware path for primitive types such as `unsigned int`.
/// \tparam TileSize The number of threads in a tiled_partition; must equal the warp size.
template<unsigned int TileSize>
__global__ void scan_kernel(const unsigned int* d_input,
                            unsigned int*       d_inclusive,
                            unsigned int*       d_exclusive,
                            const unsigned int  size)
{
    // The group consisting of all threads in the block.
    cg::thread_block block = cg::this_thread_block();

    // Partition the block into warp-sized tiles. The partition size must be known at
    // compile time, which is why the kernel is templated on it.
    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(block);

    // Global index of this thread.
    const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load this thread's element. Out-of-bounds lanes contribute the additive identity (0)
    // so that every lane of the tile participates in the collective scan below.
    const unsigned int val = global_id < size ? d_input[global_id] : 0u;

    // Inclusive scan over the tile: each lane receives the sum of all elements up to and
    // including its own. `plus` is used by default.
    const unsigned int inc = cg::inclusive_scan(tile, val);

    // Exclusive scan over the tile: each lane receives the sum of all preceding elements.
    // For `plus` the identity is 0 on both AMD and NVIDIA, so lane 0 correctly returns 0.
    const unsigned int exc = cg::exclusive_scan(tile, val);

    // Write the results back for in-bounds threads.
    if(global_id < size)
    {
        d_inclusive[global_id] = inc;
        d_exclusive[global_id] = exc;
    }
}

/// \brief Runs the prefix sum example for a tile size equal to the device warp size.
/// \tparam TileSize The warp size the device reports (32 on RDNA, 64 on CDNA).
template<unsigned int TileSize>
int run_scan()
{
    // Number of warp-sized tiles per block.
    constexpr unsigned int tiles_per_block = 4;

    // Number of threads per block. Each tile scans TileSize contiguous elements.
    constexpr unsigned int block_size = tiles_per_block * TileSize;

    // Number of blocks to launch.
    constexpr unsigned int num_blocks = 3;

    // Total number of elements. The array is partitioned into independent, warp-sized
    // segments, one per tile.
    constexpr unsigned int size = num_blocks * block_size;

    // Number of tiles the input is divided into.
    constexpr unsigned int num_tiles = size / TileSize;

    // Total size (in bytes) of the input and output arrays.
    constexpr size_t size_bytes = sizeof(unsigned int) * size;

    // Allocate and initialize the host input with a simple deterministic pattern. Keeping
    // the values small ensures the per-tile sums do not overflow an unsigned int.
    std::vector<unsigned int> h_input(size);
    for(unsigned int i = 0; i < size; i++)
    {
        h_input[i] = i + 1;
    }

    // Allocate the host output vectors.
    std::vector<unsigned int> h_inclusive(size);
    std::vector<unsigned int> h_exclusive(size);

    // Allocate device memory for the input and outputs.
    unsigned int* d_input{};
    unsigned int* d_inclusive{};
    unsigned int* d_exclusive{};
    HIP_CHECK(hipMalloc(&d_input, size_bytes));
    HIP_CHECK(hipMalloc(&d_inclusive, size_bytes));
    HIP_CHECK(hipMalloc(&d_exclusive, size_bytes));

    // Transfer the input to the device memory.
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), size_bytes, hipMemcpyHostToDevice));

    // Launch the scan kernel. A normal launch is sufficient because the scans are
    // intra-tile and require no grid-wide synchronization.
    scan_kernel<TileSize><<<num_blocks, block_size>>>(d_input, d_inclusive, d_exclusive, size);

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Transfer the results back to the host.
    HIP_CHECK(hipMemcpy(h_inclusive.data(), d_inclusive, size_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_exclusive.data(), d_exclusive, size_bytes, hipMemcpyDeviceToHost));

    // Free the resources on the device.
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_inclusive));
    HIP_CHECK(hipFree(d_exclusive));

    // Perform the reference (CPU) calculation per tile-segment and compare it element by
    // element with the device results.
    unsigned int errors = 0;
    for(unsigned int t = 0; t < num_tiles; t++)
    {
        unsigned int running_sum = 0;
        for(unsigned int j = 0; j < TileSize; j++)
        {
            const unsigned int idx = t * TileSize + j;

            // Exclusive prefix sum: the sum of all preceding elements in the tile.
            const unsigned int ref_exclusive = running_sum;

            // Inclusive prefix sum: the exclusive prefix sum plus this element.
            running_sum += h_input[idx];
            const unsigned int ref_inclusive = running_sum;

            errors += (h_inclusive[idx] != ref_inclusive);
            errors += (h_exclusive[idx] != ref_exclusive);
        }
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

    return 0;
}

int main()
{
    // Query the warp size of the current device.
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    const unsigned int warp_size = static_cast<unsigned int>(prop.warpSize);

    // The tiled_partition size must be known at compile time, so dispatch to a kernel
    // instantiated at the device warp size. Only warp sizes of 32 (RDNA) and 64 (CDNA)
    // are handled here.
    if(warp_size == 64)
    {
        return run_scan<64>();
    }
    else if(warp_size == 32)
    {
        return run_scan<32>();
    }
    else
    {
        std::cout << "Unsupported warp size " << warp_size
                  << ". This example supports warp sizes of 32 or 64." << std::endl;
        return 0;
    }
}
