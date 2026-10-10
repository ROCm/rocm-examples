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

#ifndef BATCH_NTT_HPP
#define BATCH_NTT_HPP

#include "ntt.hpp"
#include "params.h"
#include "reduce.hpp"
#include <hip/hip_runtime.h>
#include <stdint.h>

#define SP(i) ((i) + ((i) >> 5))
#define SPAD (PARAM_N + (PARAM_N >> 5)) /* 264 */

__global__ void batch_ntt_kernel(int16_t* __restrict__ polys, int batch_count)
{
    int poly_idx = blockIdx.x;
    if(poly_idx >= batch_count)
        return;

    int tid = (int)threadIdx.x; /* 0..127 */

    __shared__ int16_t s[SPAD];

    int16_t* base    = polys + poly_idx * PARAM_N;
    s[SP(tid)]       = base[tid];
    s[SP(tid + 128)] = base[tid + 128];
    __syncthreads();

#if ALGORITHM == ALGO_KYBER

    /* Level 7: len=128, 1 group */
    {
        int16_t zeta   = ntt_zetas[1];
        int     j      = tid; /* 0..127 */
        int16_t t      = fqmul(zeta, s[SP(j + 128)]);
        s[SP(j + 128)] = s[SP(j)] - t;
        s[SP(j)]       = s[SP(j)] + t;
    }
    __syncthreads();

    /* Level 6: len=64, 2 groups, zeta[2,3] */
    {
        int     group        = tid >> 6; /* 0 or 1 */
        int     lane         = tid & 0x3F; /* 0..63 */
        int16_t zeta         = ntt_zetas[2 + group];
        int     base_idx     = group * 128 + lane;
        int16_t t            = fqmul(zeta, s[SP(base_idx + 64)]);
        s[SP(base_idx + 64)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 5: len=32, 4 groups, zeta[4..7] */
    {
        int     group        = tid >> 5;
        int     lane         = tid & 0x1F;
        int16_t zeta         = ntt_zetas[4 + group];
        int     base_idx     = group * 64 + lane;
        int16_t t            = fqmul(zeta, s[SP(base_idx + 32)]);
        s[SP(base_idx + 32)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 4: len=16, 8 groups, zeta[8..15] */
    {
        int     group        = tid >> 4;
        int     lane         = tid & 0x0F;
        int16_t zeta         = ntt_zetas[8 + group];
        int     base_idx     = group * 32 + lane;
        int16_t t            = fqmul(zeta, s[SP(base_idx + 16)]);
        s[SP(base_idx + 16)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 3: len=8, 16 groups, zeta[16..31] */
    {
        int     group       = tid >> 3;
        int     lane        = tid & 0x07;
        int16_t zeta        = ntt_zetas[16 + group];
        int     base_idx    = group * 16 + lane;
        int16_t t           = fqmul(zeta, s[SP(base_idx + 8)]);
        s[SP(base_idx + 8)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 2: len=4, 32 groups, zeta[32..63] */
    {
        int     group       = tid >> 2;
        int     lane        = tid & 0x03;
        int16_t zeta        = ntt_zetas[32 + group];
        int     base_idx    = group * 8 + lane;
        int16_t t           = fqmul(zeta, s[SP(base_idx + 4)]);
        s[SP(base_idx + 4)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 1: len=2, 64 groups, zeta[64..127] */
    {
        int     group       = tid >> 1;
        int     lane        = tid & 0x01;
        int16_t zeta        = ntt_zetas[64 + group];
        int     base_idx    = group * 4 + lane;
        int16_t t           = fqmul(zeta, s[SP(base_idx + 2)]);
        s[SP(base_idx + 2)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

#elif ALGORITHM == ALGO_AIGIS_ENC

    /* Level 7: len=128, 1 group, zeta[1] */
    {
        int16_t zeta   = ntt_zetas[1];
        int     j      = tid;
        int16_t t      = fqmul(zeta, s[SP(j + 128)]);
        s[SP(j + 128)] = s[SP(j)] - t;
        s[SP(j)]       = s[SP(j)] + t;
    }
    __syncthreads();

    /* Level 6: len=64, 2 groups */
    {
        int     group        = tid >> 6;
        int     lane         = tid & 0x3F;
        int16_t zeta         = ntt_zetas[2 + group];
        int     base_idx     = group * 128 + lane;
        int16_t t            = fqmul(zeta, s[SP(base_idx + 64)]);
        s[SP(base_idx + 64)] = barrett_reduce((int16_t)(s[SP(base_idx)] - t));
        s[SP(base_idx)]      = barrett_reduce((int16_t)(s[SP(base_idx)] + t));
    }
    __syncthreads();

    /* Level 5: len=32, 4 groups */
    {
        int     group        = tid >> 5;
        int     lane         = tid & 0x1F;
        int16_t zeta         = ntt_zetas[4 + group];
        int     base_idx     = group * 64 + lane;
        int16_t t            = fqmul(zeta, s[SP(base_idx + 32)]);
        s[SP(base_idx + 32)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 4: len=16, 8 groups */
    {
        int     group        = tid >> 4;
        int     lane         = tid & 0x0F;
        int16_t zeta         = ntt_zetas[8 + group];
        int     base_idx     = group * 32 + lane;
        int16_t t            = fqmul(zeta, s[SP(base_idx + 16)]);
        s[SP(base_idx + 16)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 3: len=8, 16 groups */
    {
        int     group       = tid >> 3;
        int     lane        = tid & 0x07;
        int16_t zeta        = ntt_zetas[16 + group];
        int     base_idx    = group * 16 + lane;
        int16_t t           = fqmul(zeta, s[SP(base_idx + 8)]);
        s[SP(base_idx + 8)] = barrett_reduce((int16_t)(s[SP(base_idx)] - t));
        s[SP(base_idx)]     = barrett_reduce((int16_t)(s[SP(base_idx)] + t));
    }
    __syncthreads();

    /* Level 2: len=4, 32 groups */
    {
        int     group       = tid >> 2;
        int     lane        = tid & 0x03;
        int16_t zeta        = ntt_zetas[32 + group];
        int     base_idx    = group * 8 + lane;
        int16_t t           = fqmul(zeta, s[SP(base_idx + 4)]);
        s[SP(base_idx + 4)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 1: len=2, 64 groups */
    {
        int     group       = tid >> 1;
        int     lane        = tid & 0x01;
        int16_t zeta        = ntt_zetas[64 + group];
        int     base_idx    = group * 4 + lane;
        int16_t t           = fqmul(zeta, s[SP(base_idx + 2)]);
        s[SP(base_idx + 2)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 0: len=1, 128 groups */
    {
        int     group       = tid;
        int16_t zeta        = ntt_zetas[128 + group];
        int     base_idx    = group * 2;
        int16_t t           = fqmul(zeta, s[SP(base_idx + 1)]);
        s[SP(base_idx + 1)] = barrett_reduce((int16_t)(s[SP(base_idx)] - t));
        s[SP(base_idx)]     = barrett_reduce((int16_t)(s[SP(base_idx)] + t));
    }
    __syncthreads();

#endif /* ALGORITHM for NTT levels */

    base[tid]       = s[SP(tid)];
    base[tid + 128] = s[SP(tid + 128)];
}

__global__ void batch_invntt_kernel(int16_t* __restrict__ polys, int batch_count)
{
    int poly_idx = blockIdx.x;
    if(poly_idx >= batch_count)
        return;

    int tid = (int)threadIdx.x;

    __shared__ int16_t s[SPAD];

    int16_t* base    = polys + poly_idx * PARAM_N;
    s[SP(tid)]       = base[tid];
    s[SP(tid + 128)] = base[tid + 128];
    __syncthreads();

#if ALGORITHM == ALGO_KYBER

    {
        int     group       = tid >> 1;
        int     lane        = tid & 0x01;
        int16_t zeta        = ntt_zetas[64 + group];
        int     base_idx    = group * 4 + lane;
        int16_t t           = s[SP(base_idx)];
        s[SP(base_idx)]     = barrett_reduce((int16_t)(t + s[SP(base_idx + 2)]));
        s[SP(base_idx + 2)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 2)] - t));
    }
    __syncthreads();

    {
        int     group       = tid >> 2;
        int     lane        = tid & 0x03;
        int16_t zeta        = ntt_zetas[32 + group];
        int     base_idx    = group * 8 + lane;
        int16_t t           = s[SP(base_idx)];
        s[SP(base_idx)]     = barrett_reduce((int16_t)(t + s[SP(base_idx + 4)]));
        s[SP(base_idx + 4)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 4)] - t));
    }
    __syncthreads();

    {
        int     group       = tid >> 3;
        int     lane        = tid & 0x07;
        int16_t zeta        = ntt_zetas[16 + group];
        int     base_idx    = group * 16 + lane;
        int16_t t           = s[SP(base_idx)];
        s[SP(base_idx)]     = barrett_reduce((int16_t)(t + s[SP(base_idx + 8)]));
        s[SP(base_idx + 8)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 8)] - t));
    }
    __syncthreads();

    {
        int     group        = tid >> 4;
        int     lane         = tid & 0x0F;
        int16_t zeta         = ntt_zetas[8 + group];
        int     base_idx     = group * 32 + lane;
        int16_t t            = s[SP(base_idx)];
        s[SP(base_idx)]      = barrett_reduce((int16_t)(t + s[SP(base_idx + 16)]));
        s[SP(base_idx + 16)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 16)] - t));
    }
    __syncthreads();

    {
        int     group        = tid >> 5;
        int     lane         = tid & 0x1F;
        int16_t zeta         = ntt_zetas[4 + group];
        int     base_idx     = group * 64 + lane;
        int16_t t            = s[SP(base_idx)];
        s[SP(base_idx)]      = barrett_reduce((int16_t)(t + s[SP(base_idx + 32)]));
        s[SP(base_idx + 32)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 32)] - t));
    }
    __syncthreads();

    {
        int     group        = tid >> 6;
        int     lane         = tid & 0x3F;
        int16_t zeta         = ntt_zetas[2 + group];
        int     base_idx     = group * 128 + lane;
        int16_t t            = s[SP(base_idx)];
        s[SP(base_idx)]      = barrett_reduce((int16_t)(t + s[SP(base_idx + 64)]));
        s[SP(base_idx + 64)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 64)] - t));
    }
    __syncthreads();

    {
        int16_t zeta   = ntt_zetas[1];
        int     j      = tid;
        int16_t t      = s[SP(j)];
        s[SP(j)]       = barrett_reduce((int16_t)(t + s[SP(j + 128)]));
        s[SP(j + 128)] = fqmul(zeta, (int16_t)(s[SP(j + 128)] - t));
    }
    __syncthreads();

    {
        const int16_t f  = 1441;
        s[SP(tid)]       = fqmul(s[SP(tid)], f);
        s[SP(tid + 128)] = fqmul(s[SP(tid + 128)], f);
    }
    __syncthreads();

#elif ALGORITHM == ALGO_AIGIS_ENC

    /* Level 0: len=1, 128 groups */
    {
        int     group    = tid;
        int32_t zeta     = ntt_zetas_inv[group];
        int     base_idx = group * 2;
        int32_t t        = s[SP(base_idx)];
        s[SP(base_idx)]  = (int16_t)(t + s[SP(base_idx + 1)]);
        t -= s[SP(base_idx + 1)];
        s[SP(base_idx + 1)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 1: len=2, 64 groups, Barrett */
    {
        int     group    = tid >> 1;
        int     lane     = tid & 0x01;
        int32_t zeta     = ntt_zetas_inv[128 + group];
        int     base_idx = group * 4 + lane;
        int32_t t        = s[SP(base_idx)];
        s[SP(base_idx)]  = barrett_reduce((int16_t)(t + s[SP(base_idx + 2)]));
        t -= s[SP(base_idx + 2)];
        s[SP(base_idx + 2)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 2: len=4, 32 groups */
    {
        int     group    = tid >> 2;
        int     lane     = tid & 0x03;
        int32_t zeta     = ntt_zetas_inv[192 + group];
        int     base_idx = group * 8 + lane;
        int32_t t        = s[SP(base_idx)];
        s[SP(base_idx)]  = (int16_t)(t + s[SP(base_idx + 4)]);
        t -= s[SP(base_idx + 4)];
        s[SP(base_idx + 4)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 3: len=8, 16 groups, Barrett */
    {
        int     group    = tid >> 3;
        int     lane     = tid & 0x07;
        int32_t zeta     = ntt_zetas_inv[224 + group];
        int     base_idx = group * 16 + lane;
        int32_t t        = s[SP(base_idx)];
        s[SP(base_idx)]  = barrett_reduce((int16_t)(t + s[SP(base_idx + 8)]));
        t -= s[SP(base_idx + 8)];
        s[SP(base_idx + 8)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 4: len=16, 8 groups */
    {
        int     group    = tid >> 4;
        int     lane     = tid & 0x0F;
        int32_t zeta     = ntt_zetas_inv[240 + group];
        int     base_idx = group * 32 + lane;
        int32_t t        = s[SP(base_idx)];
        s[SP(base_idx)]  = (int16_t)(t + s[SP(base_idx + 16)]);
        t -= s[SP(base_idx + 16)];
        s[SP(base_idx + 16)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 5: len=32, 4 groups, Barrett */
    {
        int     group    = tid >> 5;
        int     lane     = tid & 0x1F;
        int32_t zeta     = ntt_zetas_inv[248 + group];
        int     base_idx = group * 64 + lane;
        int32_t t        = s[SP(base_idx)];
        s[SP(base_idx)]  = barrett_reduce((int16_t)(t + s[SP(base_idx + 32)]));
        t -= s[SP(base_idx + 32)];
        s[SP(base_idx + 32)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 6: len=64, 2 groups */
    {
        int     group    = tid >> 6;
        int     lane     = tid & 0x3F;
        int32_t zeta     = ntt_zetas_inv[252 + group];
        int     base_idx = group * 128 + lane;
        int32_t t        = s[SP(base_idx)];
        s[SP(base_idx)]  = (int16_t)(t + s[SP(base_idx + 64)]);
        t -= s[SP(base_idx + 64)];
        s[SP(base_idx + 64)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    {
        int32_t zeta = ntt_zetas_inv[254];
        int     j    = tid;
        int32_t t    = s[SP(j)];
        /* r[j] = (r[j] + r[j+128]) * N^{-1} mod Q */
        s[SP(j)] = montgomery_reduce(256 * (t + s[SP(j + 128)]));
        t -= s[SP(j + 128)];
        s[SP(j + 128)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

#endif /* ALGORITHM for INVNTT */

    base[tid]       = s[SP(tid)];
    base[tid + 128] = s[SP(tid + 128)];
}

static inline void launch_batch_ntt(int16_t* d_polys, int batch_count, hipStream_t stream = 0)
{
    batch_ntt_kernel<<<batch_count, 128, 0, stream>>>(d_polys, batch_count);
}

static inline void launch_batch_invntt(int16_t* d_polys, int batch_count, hipStream_t stream = 0)
{
    batch_invntt_kernel<<<batch_count, 128, 0, stream>>>(d_polys, batch_count);
}

#undef SP
#undef SPAD

#endif /* BATCH_NTT_HPP */
