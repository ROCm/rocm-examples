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
__global__ void weighting_kernel(float* p, std::size_t pitch,
                                 std::uint32_t N_h, std::uint32_t N_v,
                                 float d_sd, float d_so,
                                 float h_min, float v_min,
                                 float d_h, float d_v)
{
    auto const h = blockDim.x * blockIdx.x + threadIdx.x;
    auto const v = blockDim.y * blockIdx.y + threadIdx.y;

    if((h < N_h) && (v < N_v))
    {
        auto row = reinterpret_cast<float*>(reinterpret_cast<char*>(p) + v * pitch);
        
        // physical detector coordinates in mm (pixel center); adding 0.5f to stay consistent with texture fetching
        // during backprojection
        auto const h_j = (h + 0.5f) * d_h + h_min;
        auto const v_i = (v + 0.5f) * d_v + v_min;

        // distance weighting
        auto const dist_w = rsqrtf(d_sd * d_sd + h_j * h_j + v_i * v_i);

        // cosine weighting
        auto const cos_w = d_so * rsqrtf(d_so * d_so + h_j * h_j + v_i * v_i);

        // calculate weight
        auto const w_ij = d_sd * dist_w * cos_w;

        // write value
        row[h] *= w_ij;
    }
}
