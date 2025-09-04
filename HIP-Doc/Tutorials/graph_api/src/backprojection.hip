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

#include "backprojection.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

__forceinline__ __device__ auto vol_centered_coordinate(unsigned int coord, std::uint32_t dim, float size) -> float
{
    return (static_cast<float>(coord) - 0.5f * (static_cast<float>(dim) - 1.f)) * size;
}

__forceinline__ __device__ auto proj_pixel_coordinate(float coord, float minimum, float px_size) -> float
{
    return (coord - minimum) / px_size;
}

__global__ void backprojection_kernel(float* __restrict__ vol, std::size_t pitch,
                                      std::size_t N_x, std::size_t N_y, std::size_t N_z,
                                      std::size_t s_N_z, std::size_t z_offset,
                                      float d_x, float d_y, float d_z,
                                      hipTextureObject_t proj, float h_min, float v_min,
                                      float sin_theta, float cos_theta, float d_h, float d_v,
                                      float d_sd, float d_so)
{
    auto const x_in = blockIdx.x * blockDim.x + threadIdx.x;
    auto const y_in = blockIdx.y * blockDim.y + threadIdx.y;
    auto z_in = blockIdx.z * blockDim.z + threadIdx.z;

    if((x_in < N_x) && (y_in < N_y) && (z_in < s_N_z))
    {
        auto const slice_pitch = pitch * N_y;
        auto slice = reinterpret_cast<char*>(vol) + z_in * slice_pitch;
        auto row = reinterpret_cast<float*>(slice + y_in * pitch);

        // load old value from global memory while executing other instructions
        auto old_val = row[x_in];

        // add offset for current subvolume
        z_in += z_offset;

        // get coordinates relative to volume center (0, 0, 0)
        auto x = vol_centered_coordinate(x_in, N_x, d_x);
        auto y = vol_centered_coordinate(y_in, N_y, d_y);
        auto z = vol_centered_coordinate(z_in, N_z, d_z);

        // rotate coordinates
        auto s = x * cos_theta + y * sin_theta;
        auto t = -x * sin_theta + y * cos_theta;

        // project rotated coordinates onto detector; add 0.5 to each coordinate to reflect HIP's filtering mechanism
        auto factor = d_sd / (s + d_so);
        auto h = proj_pixel_coordinate(t * factor, h_min, d_h) + 0.5f;
        auto v = proj_pixel_coordinate(z * factor, v_min, d_v) + 0.5f;

        // get projection value (with implicit linear interpolation)
        auto det = tex2D<float>(proj, h, v);

        // backproject and write value
        auto u = -(d_so / (s + d_so));
        row[x_in] = old_val + 0.5f * det * u * u;
    }
}