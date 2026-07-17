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

#include <hip/cooperative_groups/memcpy_async.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#include <cstddef>
#include <cstdlib>
#include <utility>

/// \brief Number of `float` elements in a single tile.
/// The tile size is also used as the block size, so every thread of the block
/// is responsible for exactly one element per tile.
constexpr unsigned int tile_size = 256;

/// \brief Number of tiles streamed through the double-buffered pipeline.
constexpr unsigned int num_tiles = 8;

/// \brief Total number of elements of the input and output arrays.
constexpr unsigned int size = num_tiles * tile_size;

/// \brief Scale factor of the element-wise operation `out[i] = scale * in[i] + bias`.
constexpr float scale = 2.0f;

/// \brief Bias term of the element-wise operation `out[i] = scale * in[i] + bias`.
constexpr float bias = 3.0f;

/// \brief Double-buffered tile pipeline over a 1D array.
///
/// The kernel streams `num_tiles` tiles of `tile_size` elements through two LDS
/// (shared memory) buffers. While one buffer is being consumed, the next tile is
/// prefetched into the other buffer. A split barrier (`barrier_arrive` /
/// `barrier_wait`) separates the point where a thread has finished reading a
/// buffer from the point where the block guarantees that every thread is done,
/// which is what makes reusing the two buffers race-free.
///
/// Note: `cooperative_groups::memcpy_async` is an asynchronous, group-collective
/// copy (typically global <-> LDS). HIP does not expose a separate wait handle
/// (there is no `cg::wait()`), so its completion must be enforced by a following
/// group barrier. Here the copy of the next tile is issued *before*
/// `barrier_arrive`, so the release fence of `barrier_arrive` and the acquire
/// fence of `barrier_wait` order the copy's completion and make the prefetched
/// buffer visible to every thread by the next iteration. The split barrier also
/// lets the current tile's computation run as independent work between
/// `barrier_arrive` and `barrier_wait`.
__global__ void double_buffered_tile_kernel(const float* in, float* out)
{
    namespace cg = cooperative_groups;

    // The group used for the collective copies and the split barrier: every
    // thread of the block participates.
    cg::thread_block block = cg::this_thread_block();

    // The two LDS buffers that are alternated between iterations.
    __shared__ float buf[2][tile_size];

    // Rank of this thread inside the block; also the element index within a tile.
    const unsigned int tid = block.thread_rank();

    // Load the first tile into buf[0]. `memcpy_async` is an asynchronous,
    // collective call: every thread of the group must call it. Its completion is
    // enforced here with a block-wide barrier before the tile is consumed. In
    // this configuration each thread copies exactly the element it later reads.
    cg::memcpy_async(block, buf[0], in + 0, tile_size * sizeof(float));

    // Complete the async load and make the tile visible to every thread.
    block.sync();

    for(unsigned int t = 0; t < num_tiles; ++t)
    {
        // Index of the buffer holding the tile consumed during this iteration.
        const unsigned int cur = t & 1u;

        // Kick off the async load of the next tile into the other buffer. It is
        // issued *before* `barrier_arrive` so that the release fence in
        // `barrier_arrive` orders its completion and the acquire fence in
        // `barrier_wait` makes it visible. The target buf[(t + 1) & 1] is never
        // the buffer being consumed (buf[cur]), so the load and the reads below
        // target disjoint memory.
        if(t + 1 < num_tiles)
        {
            cg::memcpy_async(block,
                             buf[(t + 1) & 1u],
                             in + (t + 1) * tile_size,
                             tile_size * sizeof(float));
        }

        // Split barrier, part 1: `barrier_arrive` publishes the prefetch (release
        // fence) and returns a token without blocking, opening a window for
        // independent work.
        auto token = block.barrier_arrive();

        // Independent work performed while the barrier is in progress: consume
        // the current tile (already valid) and write this tile's results. This
        // reads buf[cur] only, so it is independent of the in-flight prefetch
        // into the other buffer.
        const unsigned int global_index = t * tile_size + tid;
        out[global_index]               = scale * buf[cur][tid] + bias;

        // Split barrier, part 2: `barrier_wait` completes the barrier (block
        // barrier + acquire fence). After it:
        // - the prefetch into the other buffer is complete and visible, ready to
        //   be consumed by the next iteration, and
        // - every thread has finished reading buf[cur], so it may safely be
        //   overwritten later.
        //
        // Race-freedom argument:
        // - The prefetch targets the *other* buffer, buf[(t + 1) & 1], which is
        //   disjoint from buf[cur] read as independent work this iteration.
        // - The prefetch is issued before `barrier_arrive`, so the barrier's
        //   release/acquire fences order its completion and visibility: when the
        //   next iteration consumes that buffer, the copy is guaranteed done.
        // - buf[cur] is only overwritten by the prefetch issued in the *next*
        //   iteration (whose target buf[((t + 1) + 1) & 1] == buf[cur]). That
        //   prefetch cannot begin until every thread has passed this iteration's
        //   `barrier_wait`, i.e. after every thread finished reading buf[cur].
        block.barrier_wait(std::move(token));
    }
}

/// \brief Host reference: applies the same element-wise operation over the array.
std::vector<float> ref_transform(const std::vector<float>& input)
{
    std::vector<float> result(input.size());
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        result[i] = scale * input[i] + bias;
    }
    return result;
}

int main()
{
    // Total size (in bytes) of the input and output arrays.
    constexpr std::size_t size_bytes = sizeof(float) * size;

    // The tile size is used as the block size, so every thread handles exactly
    // one element per tile.
    static_assert(size == num_tiles * tile_size, "size must equal num_tiles * tile_size");

    // Set up the input data on the host.
    std::vector<float> h_in(size);
    for(unsigned int i = 0; i < size; ++i)
    {
        h_in[i] = static_cast<float>(i);
    }
    std::vector<float> h_out(size);

    // Allocate device memory for the input and output arrays.
    float* d_in{};
    float* d_out{};
    HIP_CHECK(hipMalloc(&d_in, size_bytes));
    HIP_CHECK(hipMalloc(&d_out, size_bytes));

    // Transfer the input array to the device.
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), size_bytes, hipMemcpyHostToDevice));

    // Launch the double-buffered pipeline in a single block. No cooperative
    // launch is required because the block-level split barrier and the
    // block-collective copies only synchronize threads within one block.
    double_buffered_tile_kernel<<<dim3(1), dim3(tile_size), 0, hipStreamDefault>>>(d_in, d_out);

    // Check that the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Transfer the result back to the host.
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, size_bytes, hipMemcpyDeviceToHost));

    // Free the resources on the device.
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    // Perform the reference (CPU) calculation.
    const std::vector<float> ref = ref_transform(h_in);

    // Check the results' validity. The chosen constants and inputs make every
    // result exactly representable as a `float`, so an exact comparison is used.
    unsigned int errors{};
    for(unsigned int i = 0; i < size; ++i)
    {
        errors += (h_out[i] != ref[i]);
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
