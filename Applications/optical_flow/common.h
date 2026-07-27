// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// HIP hipResourceTypePitch2D requires pitchInBytes to be a multiple of 256 bytes.
// 64 floats * 4 bytes = 256 bytes satisfies this
constexpr int STRIDE_ALIGNMENT = 64;

inline int iAlignUp(int n, int m = STRIDE_ALIGNMENT)
{
    int mod = n % m;
    if (mod)
        return n + m - mod;
    else
        return n;
}

inline int iDivUp(int n, int m) { return (n + m - 1) / m; }

template <typename T> inline void Swap(T &a, T &b)
{
    T t = a;
    a   = b;
    b   = t;
}

// Software bilinear sampler used on architectures that lack hardware texture
// units (e.g. CDNA / MI300X). Replicates the behaviour of tex2D<float> with
// normalizedCoords=true, hipFilterModeLinear, hipAddressModeMirror on both
// axes. Guarded by __HIPCC__ so plain C++ translation units do not see the
// __device__ keyword. Always compiled for all GPU targets so that *KernelSW
// variants are available regardless of device architecture.
#ifdef __HIPCC__

__device__ inline float tex2D_mirror_coord(float u, int N)
{
    // Fold into [0, 2) then reflect the upper half back to [0, 1].
    u = fabsf(u);
    u = u - floorf(u * 0.5f) * 2.0f; // u in [0, 2)
    if (u > 1.0f)
        u = 2.0f - u; // u in [0, 1]
    // Convert to continuous texel space; clamp for floating-point edge cases.
    float p = u * (float)N - 0.5f;
    if (p < 0.0f)
        p = 0.0f;
    if (p > (float)(N - 1))
        p = (float)(N - 1);
    return p;
}

__device__ inline float tex2D_bilinear(
    const float* data, int width, int height, int stride, float u, float v)
{
    float px = tex2D_mirror_coord(u, width);
    float py = tex2D_mirror_coord(v, height);

    int x0 = (int)floorf(px);
    int y0 = (int)floorf(py);
    int x1 = x0 + 1 < width  ? x0 + 1 : width  - 1;
    int y1 = y0 + 1 < height ? y0 + 1 : height - 1;

    float ax = px - (float)x0;
    float ay = py - (float)y0;

    float v00 = data[y0 * stride + x0];
    float v10 = data[y0 * stride + x1];
    float v01 = data[y1 * stride + x0];
    float v11 = data[y1 * stride + x1];

    return (1.0f - ay) * ((1.0f - ax) * v00 + ax * v10)
           + ay * ((1.0f - ax) * v01 + ax * v11);
}

#endif // __HIPCC__
