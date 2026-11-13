// MIT License
//
// Copyright (c) 2026 firedoil
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

#ifndef REDUCE_HPP
#define REDUCE_HPP

#include "params.h"
#include <stdint.h>

static __device__ __forceinline__ int16_t montgomery_reduce(int32_t a)
{
    int16_t t = (int16_t)((int16_t)a * (int16_t)PARAM_QINV);
    return (int16_t)((a - (int32_t)t * PARAM_Q) >> 16);
}

static __device__ __forceinline__ int16_t fqmul(int16_t a, int16_t b)
{
    return montgomery_reduce((int32_t)a * b);
}

static __device__ __forceinline__ int16_t barrett_reduce(int16_t a)
{
#if ALGORITHM == ALGO_KYBER
    /* Kyber Q=3329, v=(2^26 + Q/2)/Q = 20159 */
    const int16_t v = (int16_t)(((1 << 26) + PARAM_Q / 2) / PARAM_Q);
    int16_t       t = (int16_t)(((int32_t)v * a + (1 << 25)) >> 26);
    return a - t * (int16_t)PARAM_Q;
#elif ALGORITHM == ALGO_AIGIS_ENC

    int16_t u = (int16_t)((a + (1 << 12)) >> 13);
    u *= (int16_t)PARAM_Q;
    return a - u;
#endif
}

static __device__ __forceinline__ int16_t caddq(int16_t a)
{
    return a + ((a >> 15) & (int16_t)PARAM_Q);
}

static __device__ __forceinline__ int16_t caddq2(int16_t a)
{
    int16_t r = a + ((a >> 15) & (int16_t)PARAM_Q);
    return r + ((r >> 15) & (int16_t)PARAM_Q);
}

static __device__ __forceinline__ int16_t tomont(int16_t a)
{
    return fqmul(a, (int16_t)MONT_R2);
}

#endif /* REDUCE_HPP */
