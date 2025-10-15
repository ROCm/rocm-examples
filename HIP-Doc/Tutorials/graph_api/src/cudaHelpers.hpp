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

/* CUDA's vector types (float4 etc.) don't have the standard operators defined. We are defining them here. */

#ifndef CUDAHELPERS_HPP
#define CUDAHELPERS_HPP

#ifdef __HIP_PLATFORM_NVIDIA__

#include <hip/hip_runtime.h>

__host__ __device__ __forceinline__ auto operator+(float3 firstSummand, float3 secondSummand) -> float3
{
    return float3
    {
        firstSummand.x + secondSummand.x,
        firstSummand.y + secondSummand.y,
        firstSummand.z + secondSummand.z
    };
}

__host__ __device__ __forceinline__ auto operator-(float3 minuend, float3 subtrahend) -> float3
{
    return float3
    {
        minuend.x - subtrahend.x,
        minuend.y - subtrahend.y,
        minuend.z - subtrahend.z
    };
}

__host__ __device__ __forceinline__ auto operator*(float3 vectorFactor, float scalarFactor) -> float3
{
    return float3
    {
        vectorFactor.x * scalarFactor,
        vectorFactor.y * scalarFactor,
        vectorFactor.z * scalarFactor
    };
}

__host__ __device__ __forceinline__ auto operator*(float scalarFactor, float3 vectorFactor) -> float3
{
    return vectorFactor * scalarFactor;
}

__host__ __device__ __forceinline__ auto operator/=(float3 lhs, float rhs) -> float3
{
    return float3
    {
        lhs.x / rhs,
        lhs.y / rhs,
        lhs.z / rhs
    };
}

#endif

#endif