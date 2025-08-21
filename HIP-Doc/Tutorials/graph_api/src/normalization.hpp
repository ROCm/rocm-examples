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

#ifndef NORMALIZATION_HPP
#define NORMALIZATION_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

template <std::uint16_t bits>
__global__ void normalization_kernel(std::uint16_t const* __restrict__ in, std::size_t const in_pitch,
                                     float* __restrict__ out, std::size_t const out_pitch,
                                     std::uint32_t const dim_h, std::uint32_t const dim_v)
{
    constexpr auto maximum = (1 << bits) - 1;

    for(auto v = blockIdx.y * blockDim.y + threadIdx.y; v < dim_v; v += blockDim.y * gridDim.y)
    {
        for(auto h = blockIdx.x * blockDim.x + threadIdx.x; h < dim_h; h += blockDim.x * gridDim.x)
        {
            auto in_row = reinterpret_cast<std::uint16_t const*>(reinterpret_cast<char const*>(in) + v * in_pitch);
            auto out_row = reinterpret_cast<float*>(reinterpret_cast<char*>(out) + v * out_pitch);

            // Make sure there are no garbage bits in the input
            auto val = in_row[h];
            val &= 0x0FFF;

            // Normalize and invert; we want black surroundings and a white object
            out_row[h] = 1.f - (static_cast<float>(val) / maximum);
        }
    }
}

#endif
