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

#include "weighting.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/*
 Weights each pixel in the projection with a weighting factor.
 
 Since we are using a cone-beam source, each ray that hits the detector plane traverses a different length. To account
 for differences in the absorption, each pixel must be weighted with a factor which depends on the detector geometry and
 the distances during the measurement.
*/
__global__ void weighting_kernel(float* p, std::size_t const pitch,
                                 std::uint32_t const N_h, std::uint32_t const N_v,
                                 float const d_sd,
                                 float const h_min, float const v_min,
                                 float const d_h, float const d_v)
{
    auto const h = blockDim.x * blockIdx.x + threadIdx.x;
    auto const v = blockDim.y * blockIdx.y + threadIdx.y;

    if((h < N_h) && (v < N_v))
    {
        auto row = reinterpret_cast<float*>(reinterpret_cast<char*>(p) + v * pitch);
        
        // physical detector coordinates in mm
        auto const h_j = (d_h / 2.f) + h * d_h + h_min;
        auto const v_i = (d_v / 2.f) + v * d_v + v_min;

        // calculate weight
        auto const w_ij = d_sd * rsqrtf(d_sd * d_sd + h_j * h_j + v_i * v_i);

        // write value
        row[h] *= w_ij;
    }
}
