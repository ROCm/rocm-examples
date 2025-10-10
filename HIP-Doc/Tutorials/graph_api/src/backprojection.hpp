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

#ifndef BACKPROJECTION_HPP
#define BACKPROJECTION_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

__global__ void backprojection_kernel(
    float* __restrict__ vol,
    std::size_t pitch,
    ulonglong3 volDim,
    float3 voxelDim,
    hipTextureObject_t proj,
    float2 minCoord,
    float sin_theta,
    float cos_theta,
    float2 pixelDim,
    float d_sd,
    float d_so
);

// Fallback for devices without support for texture instructions
// Overloaded kernel names are not supported by manual graph creation API
__global__ void backprojection_kernel_no_tex(
    float* __restrict__ vol,
    std::size_t volPitch,
    ulonglong3 volDim,
    float3 voxelDim,
    float const* __restrict__ proj,
    std::size_t projPitch,
    uint2 projDim,
    float2 minCoord,
    float sin_theta,
    float cos_theta,
    float2 pixelDim,
    float d_sd,
    float d_so
);

#endif
