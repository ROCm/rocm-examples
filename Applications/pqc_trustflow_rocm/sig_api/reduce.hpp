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

static __device__ __forceinline__ int32_t montgomery_reduce(int64_t a)
{
    uint32_t t = (uint32_t)(int32_t)a * MONT_QINV; /* uint32 wraparound: defined */
    return (int32_t)((a - (int64_t)t * PARAM_Q) >> 32);
}

/* ---- ML-DSA centered reduction ---- */

static __device__ __forceinline__ int32_t reduce32(int32_t a)
{
    int32_t t = (a + (1 << (PARAM_QBITS - 1))) >> PARAM_QBITS;
    return a - t * PARAM_Q;
}

static __device__ __forceinline__ int32_t caddq(int32_t a)
{
    a += (a >> 31) & PARAM_Q;
    return a;
}

static __device__ __forceinline__ int32_t freeze(int32_t a)
{
    return caddq(reduce32(a));
}

/* ---- Aigis unsigned reduction [0, Q) ---- */

static __device__ __forceinline__ int32_t freeze2q(int32_t a)
{
    a += (a >> 31) & (2 * PARAM_Q);
    a -= PARAM_Q;
    a += (a >> 31) & PARAM_Q;
    return a;
}

static __device__ __forceinline__ int32_t freeze4q(int32_t a)
{
    a += (a >> 31) & (4 * PARAM_Q);
    a -= 2 * PARAM_Q;
    a += (a >> 31) & (2 * PARAM_Q);
    a -= PARAM_Q;
    a += (a >> 31) & PARAM_Q;
    return a;
}

#if ALGORITHM == ALGO_AIGIS

static __device__ __forceinline__ int32_t barrat_reduce(int32_t a)
{

    return caddq(reduce32(a));
}
#endif /* ALGO_AIGIS */

/* Montgomery multiply: c = a * b * R^{-1} mod Q */
static __device__ __forceinline__ coeff_t coeff_fqmul(coeff_t a, coeff_t b)
{
    return montgomery_reduce((coeff2_t)a * b);
}

static __device__ __forceinline__ coeff_t coeff_sub(coeff_t a, coeff_t b)
{
#if ALGORITHM == ALGO_AIGIS

    return a + 2 * PARAM_Q - b;
#else
    return a - b;
#endif
}

static __device__ __forceinline__ coeff_t coeff_reduce(coeff_t a)
{
#if ALGORITHM == ALGO_AIGIS
    return barrat_reduce(a);
#else
    return reduce32(a);
#endif
}

static __device__ __forceinline__ coeff_t coeff_normalize(coeff_t a)
{
#if ALGORITHM == ALGO_AIGIS
    return freeze2q(a);
#else
    return caddq(a);
#endif
}

static __device__ __forceinline__ coeff_t coeff_freeze_wide(coeff_t a)
{
#if ALGORITHM == ALGO_AIGIS
    return freeze4q(a);
#else
    return caddq(reduce32(a));
#endif
}

#endif /* REDUCE_HPP */
