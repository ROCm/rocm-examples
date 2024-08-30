
// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

// Configuration constants
#define HIP_TEMPLATE_KERNEL_LAUNCH

// Reduction
#include <hip_utils.hpp>
#include <tmp_utils.hpp>

// HIP API
#include <hip/hip_runtime.h>

// STL
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <execution>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace reduction
{
template<typename T, typename F, uint32_t BlockSize, uint32_t WarpSize>
__global__ static __launch_bounds__(BlockSize) void kernel(
    T* front, T* back, F op, T zero_elem, uint32_t front_size)
{
    __shared__ T shared[BlockSize];

    auto read_global_safe = [&](const uint32_t i) { return i < front_size ? front[i] : zero_elem; };

    const uint32_t tid = threadIdx.x, bid = blockIdx.x, gid = bid * (blockDim.x * 2) + tid;

    // Read input from front buffer to shared
    shared[tid] = op(read_global_safe(gid), read_global_safe(gid + blockDim.x));
    __syncthreads();

    // Shared reduction
    tmp::static_for<BlockSize / 2, tmp::greater_than<WarpSize>, tmp::divide<2>>(
        [&]<int I>()
        {
            if(tid < I)
                shared[tid] = op(shared[tid], shared[tid + I]);
            __syncthreads();
        });
    // Warp reduction
    if(tid < WarpSize)
    {
        T res = op(shared[tid], shared[tid + WarpSize]);
        tmp::static_for<WarpSize / 2, tmp::not_equal<0>, tmp::divide<2>>(
            [&]<int Delta>() { res = op(res, __shfl_down(res, Delta)); });

        // Write result from shared to back buffer
        if(tid == 0)
            back[bid] = res;
    }
}

template<typename T, typename F>
class v7
{
public:
    v7(const F&                kernel_op_in,
       const T                 zero_elem_in,
       const std::span<size_t> input_sizes_in,
       const std::span<size_t> block_sizes_in)
        : kernel_op{kernel_op_in}
        , zero_elem{zero_elem_in}
        , input_sizes{input_sizes_in}
        , block_sizes{block_sizes_in}
    {
        // Pessimistically allocate front buffer based on the largest input and the
        // back buffer smallest reduction factor
        auto smallest_factor = *std::min_element(block_sizes_in.begin(), block_sizes_in.end());
        auto largest_size    = *std::max_element(input_sizes.begin(), input_sizes.end());
        HIP_CHECK(hipMalloc((void**)&front, sizeof(T) * largest_size));
        HIP_CHECK(hipMalloc((void**)&back, new_size(smallest_factor, largest_size) * sizeof(T)));
        origi_front = front;
        origi_back  = back;

        hipDeviceProp_t devProp;
        int             device_id = 0;
        HIP_CHECK(hipGetDevice(&device_id));
        HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
        warp_size = devProp.warpSize;
    }

    ~v7()
    {
        HIP_CHECK(hipFree(front));
        HIP_CHECK(hipFree(back));
    }

    std::tuple<T, std::chrono::duration<float, std::milli>>
        operator()(std::span<const T> input, const std::size_t block_size, const std::size_t)
    {
        auto factor = block_size * 2;
        HIP_CHECK(hipMemcpy(front, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

        auto kernel_dispatcher = [&](std::size_t step_size)
        {
            tmp::static_switch<std::array{32, 64, 128, 256, 512, 1024, 2048}>(
                block_size,
                [&]<int BlockSize>() noexcept
                {
                    tmp::static_switch<std::array{32, 64}>(
                        warp_size,
                        [&]<int WarpSize>() noexcept
                        {
                            HIP_KERNEL_NAME(kernel<T, F, BlockSize, WarpSize>)<<<
                                dim3(new_size(factor, step_size)),
                                dim3(BlockSize),
                                0,
                                hipStreamDefault>>>(front, back, kernel_op, zero_elem, step_size);
                        });
                });
        };

        hipEvent_t start, end;

        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&end));
        HIP_CHECK(hipEventRecord(start, 0));
        std::size_t curr = input.size();
        while(curr > 1)
        {
            kernel_dispatcher(curr);
            hip::check(hipGetLastError(), "hipKernelLaunchGGL");

            curr = new_size(factor, curr);
            if(curr > 1)
                std::swap(front, back);
        }
        HIP_CHECK(hipEventRecord(end, 0));
        HIP_CHECK(hipEventSynchronize(end));

        T result;
        HIP_CHECK(hipMemcpy(&result, back, sizeof(T), hipMemcpyDeviceToHost));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, end));

        HIP_CHECK(hipEventDestroy(end));
        HIP_CHECK(hipEventDestroy(start));

        front = origi_front;
        back  = origi_back;

        return {result, std::chrono::duration<float, std::milli>{elapsed_mseconds}};
    }

private:
    F                      kernel_op;
    T                      zero_elem;
    std::span<std::size_t> input_sizes, block_sizes;
    T*                     front;
    T*                     back;
    T*                     origi_front;
    T*                     origi_back;
    std::size_t            warp_size;

    std::size_t new_size(const std::size_t factor, const std::size_t actual)
    {
        return actual / factor + (actual % factor == 0 ? 0 : 1);
    }
};
} // namespace reduction
