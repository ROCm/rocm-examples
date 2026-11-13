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

#ifndef POLYVEC_HPP
#define POLYVEC_HPP

#include "ntt.hpp"
#include "params.h"
#include "poly.hpp"
#include <stdint.h>

static __device__ void polyvec_add(kem_polyvec* r, const kem_polyvec* a, const kem_polyvec* b)
{
    for(int i = 0; i < PARAM_K; i++)
        poly_add(&r->vec[i], &a->vec[i], &b->vec[i]);
}

static __device__ void polyvec_reduce(kem_polyvec* r)
{
    for(int i = 0; i < PARAM_K; i++)
        poly_reduce(&r->vec[i]);
}

static __device__ void polyvec_caddq(kem_polyvec* r)
{
    for(int i = 0; i < PARAM_K; i++)
        poly_caddq(&r->vec[i]);
}

static __device__ void polyvec_caddq2(kem_polyvec* r)
{
    for(int i = 0; i < PARAM_K; i++)
        poly_caddq2(&r->vec[i]);
}

static __device__ __noinline__ void polyvec_tobytes(uint8_t* r, const kem_polyvec* a)
{
    for(int i = 0; i < PARAM_K; i++)
        poly_tobytes(r + i * PARAM_POLYBYTES, &a->vec[i]);
}

static __device__ __noinline__ void polyvec_frombytes(kem_polyvec* r, const uint8_t* a)
{
    for(int i = 0; i < PARAM_K; i++)
        poly_frombytes(&r->vec[i], a + i * PARAM_POLYBYTES);
}

static __device__ __noinline__ void polyvec_compress9(uint8_t* r, const kem_poly* a)
{
    for(int i = 0; i < PARAM_N / 8; i++)
    {
        uint16_t c[8];
        for(int j = 0; j < 8; j++)
            c[j] = (uint16_t)((((int32_t)caddq(a->coeffs[8 * i + j]) << 9) + PARAM_Q / 2) / PARAM_Q)
                   & 0x1FF;
        r[9 * i + 0] = (uint8_t)(c[0]);
        r[9 * i + 1] = (uint8_t)((c[0] >> 8) | (c[1] << 1));
        r[9 * i + 2] = (uint8_t)((c[1] >> 7) | (c[2] << 2));
        r[9 * i + 3] = (uint8_t)((c[2] >> 6) | (c[3] << 3));
        r[9 * i + 4] = (uint8_t)((c[3] >> 5) | (c[4] << 4));
        r[9 * i + 5] = (uint8_t)((c[4] >> 4) | (c[5] << 5));
        r[9 * i + 6] = (uint8_t)((c[5] >> 3) | (c[6] << 6));
        r[9 * i + 7] = (uint8_t)((c[6] >> 2) | (c[7] << 7));
        r[9 * i + 8] = (uint8_t)((c[7] >> 1));
    }
}

static __device__ __noinline__ void polyvec_decompress9(kem_poly* r, const uint8_t* a)
{
    for(int i = 0; i < PARAM_N / 8; i++)
    {
        uint16_t c[8];
        c[0] = ((uint16_t)a[9 * i + 0]) | ((uint16_t)(a[9 * i + 1] & 0x01) << 8);
        c[1] = ((uint16_t)a[9 * i + 1] >> 1) | ((uint16_t)(a[9 * i + 2] & 0x03) << 7);
        c[2] = ((uint16_t)a[9 * i + 2] >> 2) | ((uint16_t)(a[9 * i + 3] & 0x07) << 6);
        c[3] = ((uint16_t)a[9 * i + 3] >> 3) | ((uint16_t)(a[9 * i + 4] & 0x0F) << 5);
        c[4] = ((uint16_t)a[9 * i + 4] >> 4) | ((uint16_t)(a[9 * i + 5] & 0x1F) << 4);
        c[5] = ((uint16_t)a[9 * i + 5] >> 5) | ((uint16_t)(a[9 * i + 6] & 0x3F) << 3);
        c[6] = ((uint16_t)a[9 * i + 6] >> 6) | ((uint16_t)(a[9 * i + 7] & 0x7F) << 2);
        c[7] = ((uint16_t)a[9 * i + 7] >> 7) | ((uint16_t)(a[9 * i + 8]) << 1);
        for(int j = 0; j < 8; j++)
            r->coeffs[8 * i + j] = (int16_t)(((int32_t)c[j] * PARAM_Q + 256) >> 9);
    }
}

static __device__ __noinline__ void polyvec_compress10(uint8_t* r, const kem_poly* a)
{
    for(int i = 0; i < PARAM_N / 4; i++)
    {
        uint16_t c[4];
        for(int j = 0; j < 4; j++)
            c[j]
                = (uint16_t)((((int32_t)caddq(a->coeffs[4 * i + j]) << 10) + PARAM_Q / 2) / PARAM_Q)
                  & 0x3FF;
        r[5 * i + 0] = (uint8_t)(c[0]);
        r[5 * i + 1] = (uint8_t)((c[0] >> 8) | (c[1] << 2));
        r[5 * i + 2] = (uint8_t)((c[1] >> 6) | (c[2] << 4));
        r[5 * i + 3] = (uint8_t)((c[2] >> 4) | (c[3] << 6));
        r[5 * i + 4] = (uint8_t)((c[3] >> 2));
    }
}

static __device__ __noinline__ void polyvec_decompress10(kem_poly* r, const uint8_t* a)
{
    for(int i = 0; i < PARAM_N / 4; i++)
    {
        uint16_t c[4];
        c[0] = ((uint16_t)a[5 * i + 0]) | ((uint16_t)(a[5 * i + 1] & 0x03) << 8);
        c[1] = ((uint16_t)a[5 * i + 1] >> 2) | ((uint16_t)(a[5 * i + 2] & 0x0F) << 6);
        c[2] = ((uint16_t)a[5 * i + 2] >> 4) | ((uint16_t)(a[5 * i + 3] & 0x3F) << 4);
        c[3] = ((uint16_t)a[5 * i + 3] >> 6) | ((uint16_t)(a[5 * i + 4]) << 2);
        for(int j = 0; j < 4; j++)
            r->coeffs[4 * i + j] = (int16_t)(((int32_t)c[j] * PARAM_Q + 512) >> 10);
    }
}

static __device__ __noinline__ void polyvec_compress11(uint8_t* r, const kem_poly* a)
{
    for(int i = 0; i < PARAM_N / 8; i++)
    {
        uint16_t c[8];
        for(int j = 0; j < 8; j++)
            c[j]
                = (uint16_t)((((int32_t)caddq(a->coeffs[8 * i + j]) << 11) + PARAM_Q / 2) / PARAM_Q)
                  & 0x7FF;
        r[11 * i + 0]  = (uint8_t)(c[0]);
        r[11 * i + 1]  = (uint8_t)((c[0] >> 8) | (c[1] << 3));
        r[11 * i + 2]  = (uint8_t)((c[1] >> 5) | (c[2] << 6));
        r[11 * i + 3]  = (uint8_t)((c[2] >> 2));
        r[11 * i + 4]  = (uint8_t)((c[2] >> 10) | (c[3] << 1));
        r[11 * i + 5]  = (uint8_t)((c[3] >> 7) | (c[4] << 4));
        r[11 * i + 6]  = (uint8_t)((c[4] >> 4) | (c[5] << 7));
        r[11 * i + 7]  = (uint8_t)((c[5] >> 1));
        r[11 * i + 8]  = (uint8_t)((c[5] >> 9) | (c[6] << 2));
        r[11 * i + 9]  = (uint8_t)((c[6] >> 6) | (c[7] << 5));
        r[11 * i + 10] = (uint8_t)((c[7] >> 3));
    }
}

static __device__ __noinline__ void polyvec_decompress11(kem_poly* r, const uint8_t* a)
{
    for(int i = 0; i < PARAM_N / 8; i++)
    {
        uint16_t c[8];
        c[0] = ((uint16_t)a[11 * i + 0]) | ((uint16_t)(a[11 * i + 1] & 0x07) << 8);
        c[1] = ((uint16_t)a[11 * i + 1] >> 3) | ((uint16_t)(a[11 * i + 2] & 0x3F) << 5);
        c[2] = ((uint16_t)a[11 * i + 2] >> 6) | ((uint16_t)a[11 * i + 3] << 2)
               | ((uint16_t)(a[11 * i + 4] & 0x01) << 10);
        c[3] = ((uint16_t)a[11 * i + 4] >> 1) | ((uint16_t)(a[11 * i + 5] & 0x0F) << 7);
        c[4] = ((uint16_t)a[11 * i + 5] >> 4) | ((uint16_t)(a[11 * i + 6] & 0x7F) << 4);
        c[5] = ((uint16_t)a[11 * i + 6] >> 7) | ((uint16_t)a[11 * i + 7] << 1)
               | ((uint16_t)(a[11 * i + 8] & 0x03) << 9);
        c[6] = ((uint16_t)a[11 * i + 8] >> 2) | ((uint16_t)(a[11 * i + 9] & 0x1F) << 6);
        c[7] = ((uint16_t)a[11 * i + 9] >> 5) | ((uint16_t)a[11 * i + 10] << 3);
        for(int j = 0; j < 8; j++)
            r->coeffs[8 * i + j] = (int16_t)(((int32_t)c[j] * PARAM_Q + 1024) >> 11);
    }
}

#if ALGORITHM == ALGO_KYBER
static __device__ __noinline__ void polyvec_pk_compress(uint8_t* r, const kem_polyvec* a)
{
    polyvec_tobytes(r, a);
}
static __device__ __noinline__ void polyvec_pk_decompress(kem_polyvec* r, const uint8_t* a)
{
    polyvec_frombytes(r, a);
}
#elif ALGORITHM == ALGO_AIGIS_ENC

static __device__ __noinline__ void polyvec_pk_compress(uint8_t* r, const kem_polyvec* a)
{
    for(int i = 0; i < PARAM_K; i++)
    {
        uint8_t* dst = r + i * PARAM_BITS_PK * PARAM_N / 8;
    #if PARAM_BITS_PK == 9
        polyvec_compress9(dst, &a->vec[i]);
    #elif PARAM_BITS_PK == 10
        polyvec_compress10(dst, &a->vec[i]);
    #elif PARAM_BITS_PK == 11
        polyvec_compress11(dst, &a->vec[i]);
    #endif
    }
}

static __device__ __noinline__ void polyvec_pk_decompress(kem_polyvec* r, const uint8_t* a)
{
    for(int i = 0; i < PARAM_K; i++)
    {
        const uint8_t* src = a + i * PARAM_BITS_PK * PARAM_N / 8;
    #if PARAM_BITS_PK == 9
        polyvec_decompress9(&r->vec[i], src);
    #elif PARAM_BITS_PK == 10
        polyvec_decompress10(&r->vec[i], src);
    #elif PARAM_BITS_PK == 11
        polyvec_decompress11(&r->vec[i], src);
    #endif
    }
}
#endif /* ALGORITHM for PK compress */

static __device__ __noinline__ void polyvec_ct_compress(uint8_t* r, const kem_polyvec* a)
{
    for(int i = 0; i < PARAM_K; i++)
    {
        uint8_t* dst = r + i * PARAM_BITS_C1 * PARAM_N / 8;
#if PARAM_BITS_C1 == 9
        polyvec_compress9(dst, &a->vec[i]);
#elif PARAM_BITS_C1 == 10
        polyvec_compress10(dst, &a->vec[i]);
#elif PARAM_BITS_C1 == 11
        polyvec_compress11(dst, &a->vec[i]);
#endif
    }
}

static __device__ __noinline__ void polyvec_ct_decompress(kem_polyvec* r, const uint8_t* a)
{
    for(int i = 0; i < PARAM_K; i++)
    {
        const uint8_t* src = a + i * PARAM_BITS_C1 * PARAM_N / 8;
#if PARAM_BITS_C1 == 9
        polyvec_decompress9(&r->vec[i], src);
#elif PARAM_BITS_C1 == 10
        polyvec_decompress10(&r->vec[i], src);
#elif PARAM_BITS_C1 == 11
        polyvec_decompress11(&r->vec[i], src);
#endif
    }
}

#endif /* POLYVEC_HPP */
