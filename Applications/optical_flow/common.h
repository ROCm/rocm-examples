// MIT License
//
// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

///////////////////////////////////////////////////////////////////////////////
// Header for common includes and utility functions
///////////////////////////////////////////////////////////////////////////////

#ifndef COMMON_H
#define COMMON_H

///////////////////////////////////////////////////////////////////////////////
// Common includes
///////////////////////////////////////////////////////////////////////////////

#include <hip/hip_runtime.h>
#include "example_utils.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

///////////////////////////////////////////////////////////////////////////////
// Common constants
///////////////////////////////////////////////////////////////////////////////
// HIP hipResourceTypePitch2D requires pitchInBytes to be a multiple of 256 bytes.
// 64 floats * 4 bytes = 256 bytes satisfies this
const int StrideAlignment = 64;

///////////////////////////////////////////////////////////////////////////////
// Common functions
///////////////////////////////////////////////////////////////////////////////

// Align up n to the nearest multiple of m
inline int iAlignUp(int n, int m = StrideAlignment)
{
    int mod = n % m;

    if (mod)
        return n + m - mod;
    else
        return n;
}

// round up n/m
inline int iDivUp(int n, int m) { return (n + m - 1) / m; }

// swap two values
template <typename T> inline void Swap(T &a, T &b)
{
    T t = a;
    a   = b;
    b   = t;
}
#endif
