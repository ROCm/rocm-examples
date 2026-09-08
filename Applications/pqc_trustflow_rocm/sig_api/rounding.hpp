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

#ifndef ROUNDING_HPP
#define ROUNDING_HPP

#include "params.h"
#include "reduce.hpp"
#include <stdint.h>

/* ================================================================
 *  power2round
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __forceinline__ int32_t power2round(int32_t* a0, int32_t a)
{
    int32_t a1 = (a + (1 << (PARAM_D - 1)) - 1) >> PARAM_D;
    *a0        = a - (a1 << PARAM_D);
    return a1;
}

#elif ALGORITHM == ALGO_AIGIS

static __device__ __forceinline__ int32_t power2round(int32_t* a0, int32_t a)
{
    int32_t t;
    t = a & ((1 << PARAM_D) - 1);
    t -= (1 << (PARAM_D - 1)) + 1;
    t += (t >> 31) & (1 << PARAM_D);
    t -= (1 << (PARAM_D - 1)) - 1;
    *a0 = PARAM_Q + t;
    a   = (a - t) >> PARAM_D;
    return a;
}

#endif /* power2round */

/* ================================================================
 *  decompose
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __forceinline__ int32_t decompose(int32_t* a0, int32_t a)
{
    int32_t a1;
    #if PARAM_GAMMA2 == (PARAM_Q - 1) / 32
    a1 = (a + 127) >> 7;
    a1 = (a1 * 1025 + (1 << 21)) >> 22;
    a1 &= 15;
    *a0 = a - a1 * 2 * PARAM_GAMMA2;
    *a0 -= (((PARAM_Q - 1) / 2 - *a0) >> 31) & PARAM_Q;
    #elif PARAM_GAMMA2 == (PARAM_Q - 1) / 88
    a1 = (a + 127) >> 7;
    a1 = (a1 * 11275 + (1 << 23)) >> 24;
    a1 ^= ((43 - a1) >> 31) & a1;
    *a0 = a - a1 * 2 * PARAM_GAMMA2;
    *a0 -= (((PARAM_Q - 1) / 2 - *a0) >> 31) & PARAM_Q;
    #endif
    return a1;
}

#elif ALGORITHM == ALGO_AIGIS

static __device__ __forceinline__ int32_t decompose(int32_t* a0, int32_t a)
{
    int32_t       t, u;
    const int32_t ALPHA = 2 * PARAM_GAMMA2;

    #if PARAM_Q == 2021377
    u = ((int32_t)((uint32_t)a * 3u) >> 20) + 1;
    #elif PARAM_Q == 3870721
    u = ((int32_t)((uint32_t)a * 3u) >> 21) + 1;
    #endif
    t = a - u * ALPHA;
    u -= (t >> 31) & 1;
    t += (t >> 31) & ALPHA;
    t -= ALPHA / 2 + 1;
    t += (t >> 31) & ALPHA;
    t -= ALPHA / 2 - 1;
    u += (t >> 31) & 1;
    int32_t a1 = u;
    if(a1 == N_W1)
    {
        *a0 = PARAM_Q + t - 1;
        a1  = 0;
    }
    else
    {
        *a0 = PARAM_Q + t;
    }
    return a1;
}

#endif /* decompose */

/* ================================================================
 *  make_hint / use_hint
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __forceinline__ int32_t make_hint(int32_t a0, int32_t a1)
{
    if(a0 > PARAM_GAMMA2 || a0 < -PARAM_GAMMA2 || (a0 == -PARAM_GAMMA2 && a1 != 0))
        return 1;
    return 0;
}

static __device__ __forceinline__ int32_t use_hint(int32_t a, int32_t hint)
{
    int32_t a0, a1;
    a1 = decompose(&a0, a);
    if(hint == 0)
        return a1;
    if(a0 > 0)
        return (a1 + 1 >= N_W1) ? 0 : a1 + 1;
    else
        return (a1 - 1 < 0) ? N_W1 - 1 : a1 - 1;
}

#elif ALGORITHM == ALGO_AIGIS

static __device__ __forceinline__ int32_t make_hint(int32_t a, int32_t b)
{
    int32_t t;
    return decompose(&t, a) != decompose(&t, freeze4q(a + b));
}

static __device__ __forceinline__ int32_t use_hint(int32_t a, int32_t hint)
{
    int32_t a0, a1;
    a1 = decompose(&a0, a);
    if(hint == 0)
        return a1;
    if(a0 > PARAM_Q)
        return (a1 == (PARAM_Q - 1) / (2 * PARAM_GAMMA2) - 1) ? 0 : a1 + 1;
    else
        return (a1 == 0) ? (PARAM_Q - 1) / (2 * PARAM_GAMMA2) - 1 : a1 - 1;
}

#endif /* make_hint / use_hint */

#endif /* ROUNDING_HPP */
