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

#include "filtering.hpp"

#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>

#include <cstddef>
#include <cstdint>
#include <numbers>

// Creates a Shepp-Logan filter
__global__ void filter_creation_kernel(float* __restrict__ r, std::int32_t N_hFFT, float tau)
{
    constexpr auto pi = std::numbers::pi_v<float>;
    auto const i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < static_cast<std::uint32_t>(N_hFFT))
    {
        auto const j = -((N_hFFT - 2) / 2) + static_cast<std::int32_t>(i);

        if(j == 0)
            r[i] = (1.f / 4.f) * (1.f / tau);
        else if(j % 2 == 0)
            r[i] = 0;
        else
        {
            auto const x = pi * j / N_hFFT;
            r[i] = -(1.f / (j * j * pi * pi * tau)) * (sinf(x) / x);
        }
    }
}

__global__ void filter_absolute_kernel(hipfftComplex* R, std::uint32_t N_hTrans, float tau)
{
    auto const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N_hTrans)
    {
        auto const val = R[i];
        auto const result = tau * fabsf(sqrtf((val.x * val.x) + (val.y * val.y)));
        R[i].x = result;
        R[i].y = result;
    }
}

__global__ void filter_application_kernel(hipfftComplex* __restrict__ P, std::size_t const pitch,
                                          hipfftComplex const* __restrict__ R,
                                          std::uint32_t const N_hTrans, std::uint32_t const N_v)
{
    auto const h = blockIdx.x * blockDim.x + threadIdx.x;
    auto const v = blockIdx.y * blockDim.y + threadIdx.y;

    if(h < N_hTrans && v < N_v)
    {
        auto row = reinterpret_cast<hipfftComplex*>(reinterpret_cast<char*>(P) + v * pitch);

        row[h].x *= R[h].x;
        row[h].y *= R[h].y;
    }
}

__global__ void filter_normalization_kernel(float* p, std::size_t const pitch,
                                            std::uint32_t const N_hFFT,
                                            std::uint32_t const N_h, std::uint32_t const N_v)
{
    auto const h = blockIdx.x * blockDim.x + threadIdx.x;
    auto const v = blockIdx.y * blockDim.y + threadIdx.y;

    if((h < N_h) && (v < N_v))
    {
        auto row = reinterpret_cast<float*>(reinterpret_cast<char*>(p) + v * pitch);
        row[h] /= N_hFFT;
    }
}
