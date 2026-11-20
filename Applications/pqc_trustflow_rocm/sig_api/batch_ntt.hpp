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

/* ================================================================
 * Montgomery multiply helper for batch kernels
 * ================================================================ */
static __device__ __forceinline__ coeff_t batch_fqmul_local(coeff_t a, coeff_t b)
{
    return montgomery_reduce((coeff2_t)a * b);
}

__global__ void batch_ntt_kernel(coeff_t* d_polys, int poly_count)
{
    int poly_idx = blockIdx.x;
    if(poly_idx >= poly_count)
        return;

    int tid = threadIdx.x; /* 0..127 */

    __shared__ coeff_t s[PARAM_N + (PARAM_N >> 5)];
#define SP(i) ((i) + ((i) >> 5))

    coeff_t* base    = d_polys + (size_t)poly_idx * PARAM_N;
    s[SP(tid)]       = base[tid];
    s[SP(tid + 128)] = base[tid + 128];
    __syncthreads();

    /* Loop-based NTT: 8 stages (len=128,64,32,16,8,4,2,1) */
    unsigned int k = 0;
#pragma unroll
    for(unsigned int len = 128; len >= 1; len >>= 1)
    {
        unsigned int step     = len << 1;
        unsigned int block_id = tid / len;
        unsigned int j        = block_id * step + (tid % len);

        coeff_t zeta1  = ntt_zetas[k + 1 + block_id];
        coeff_t sj     = s[SP(j)];
        coeff_t t1     = batch_fqmul_local(zeta1, s[SP(j + len)]);
        s[SP(j + len)] = sj - t1;
        s[SP(j)]       = sj + t1;

        /* 128 threads handle N/2=128 butterflies per stage.
         * For len<=64, each thread handles 2 butterflies (j and j2). */
        if(len <= 64)
        {
            unsigned int j2_block = (tid + 128) / len;
            unsigned int j2       = j2_block * step + ((tid + 128) % len);
            if(j2 + len < PARAM_N)
            {
                coeff_t zeta2   = ntt_zetas[k + 1 + j2_block];
                coeff_t sj2     = s[SP(j2)];
                coeff_t t2      = batch_fqmul_local(zeta2, s[SP(j2 + len)]);
                s[SP(j2 + len)] = sj2 - t2;
                s[SP(j2)]       = sj2 + t2;
            }
        }

        k += (PARAM_N / step);
        if(len >= 64 || len == 1)
            __syncthreads();
        else
            __syncwarp();
    }

    base[tid]       = s[SP(tid)];
    base[tid + 128] = s[SP(tid + 128)];
#undef SP
}

__global__ void batch_invntt_kernel(coeff_t* d_polys, int poly_count)
{
    int poly_idx = blockIdx.x;
    if(poly_idx >= poly_count)
        return;

    int tid = threadIdx.x;

    __shared__ coeff_t s[PARAM_N + (PARAM_N >> 5)];
#define SP(i) ((i) + ((i) >> 5))

    coeff_t* base    = d_polys + (size_t)poly_idx * PARAM_N;
    s[SP(tid)]       = base[tid];
    s[SP(tid + 128)] = base[tid + 128];
    __syncthreads();

#if ALGORITHM == ALGO_MLDSA
    {
        unsigned int k = 256;
    #pragma unroll
        for(unsigned int len = 1; len <= 128; len <<= 1)
        {
            unsigned int step     = len << 1;
            unsigned int block_id = tid / len;
            unsigned int j        = block_id * step + (tid % len);

            coeff_t zeta1  = -ntt_zetas[k - 1 - block_id];
            coeff_t t1     = s[SP(j)];
            coeff_t sjlen  = s[SP(j + len)];
            s[SP(j)]       = t1 + sjlen;
            s[SP(j + len)] = batch_fqmul_local(zeta1, t1 - sjlen);

            if(len <= 64)
            {
                unsigned int j2_block = (tid + 128) / len;
                unsigned int j2       = j2_block * step + ((tid + 128) % len);
                if(j2 + len < PARAM_N)
                {
                    coeff_t zeta2   = -ntt_zetas[k - 1 - j2_block];
                    coeff_t t2      = s[SP(j2)];
                    coeff_t sj2len  = s[SP(j2 + len)];
                    s[SP(j2)]       = t2 + sj2len;
                    s[SP(j2 + len)] = batch_fqmul_local(zeta2, t2 - sj2len);
                }
            }

            k -= (PARAM_N / step);
            if(len >= 32)
                __syncthreads();
            else
                __syncwarp();
        }
    }

    /* Scale by N^{-1} * MONT */
    {
        const coeff_t f  = INTT_F;
        s[SP(tid)]       = batch_fqmul_local(f, s[SP(tid)]);
        s[SP(tid + 128)] = batch_fqmul_local(f, s[SP(tid + 128)]);
    }
    __syncthreads();

#elif ALGORITHM == ALGO_AIGIS
    {
        unsigned int ki = 0;
    #pragma unroll
        for(unsigned int len = 1; len <= 128; len <<= 1)
        {
            unsigned int step       = len << 1;
            unsigned int num_blocks = PARAM_N / step;

            /* Each of 128 threads handles one butterfly */
            if(tid < PARAM_N / 2)
            {
                unsigned int blk_id = tid / len;
                unsigned int pos_   = tid % len;
                unsigned int j      = blk_id * step + pos_;
                coeff_t      zeta   = ntt_zetas_inv[ki + blk_id];
                coeff_t      t      = s[SP(j)];
                coeff_t      sjlen  = s[SP(j + len)];
                s[SP(j)]            = t + sjlen;
                s[SP(j + len)]      = batch_fqmul_local(zeta, t - sjlen);
            }

            ki += num_blocks;
            if(len >= 32)
                __syncthreads();
            else
                __syncwarp();
        }
    }

    /* Scale: only first N/2 coefficients */
    {
        const coeff_t f = INTT_F;
        s[SP(tid)]      = batch_fqmul_local(f, s[SP(tid)]);
        /* s[tid+128] untouched (last-stage twiddle has N^{-1} baked in) */
    }
    __syncthreads();
#endif

    base[tid]       = s[SP(tid)];
    base[tid + 128] = s[SP(tid + 128)];
#undef SP
}

/* ================================================================
 * Host launch wrappers
 * ================================================================ */
static inline void launch_batch_ntt(coeff_t* d_polys, int count, hipStream_t stream = 0)
{
    if(count <= 0)
        return;
    batch_ntt_kernel<<<count, 128, 0, stream>>>(d_polys, count);
}

static inline void launch_batch_invntt(coeff_t* d_polys, int count, hipStream_t stream = 0)
{
    if(count <= 0)
        return;
    batch_invntt_kernel<<<count, 128, 0, stream>>>(d_polys, count);
}

static __device__ __forceinline__ void ntt_warp_par(coeff_t* r, int lane)
{

    unsigned int k = 0;
#pragma unroll
    for(unsigned int len = 128; len >= 1; len >>= 1)
    {
        unsigned int step = len << 1;

        for(int b = lane; b < PARAM_N / 2; b += 32)
        {
            unsigned int blk  = b / len;
            unsigned int pos  = b % len;
            unsigned int j    = blk * step + pos;
            coeff_t      zeta = ntt_zetas[k + 1 + blk];
            coeff_t      t    = montgomery_reduce((coeff2_t)zeta * r[j + len]);
            r[j + len]        = r[j] - t;
            r[j]              = r[j] + t;
        }
        k += (PARAM_N / step);
        __syncwarp();
    }
}

static __device__ __forceinline__ void invntt_warp_par(coeff_t* r, int lane)
{
#if ALGORITHM == ALGO_MLDSA
    unsigned int k = 256;
    #pragma unroll
    for(unsigned int len = 1; len <= 128; len <<= 1)
    {
        unsigned int step = len << 1;
        for(int b = lane; b < PARAM_N / 2; b += 32)
        {
            unsigned int blk  = b / len;
            unsigned int pos  = b % len;
            unsigned int j    = blk * step + pos;
            coeff_t      zeta = -ntt_zetas[k - 1 - blk];
            coeff_t      t    = r[j];
            r[j]              = t + r[j + len];
            r[j + len]        = montgomery_reduce((coeff2_t)zeta * (t - r[j + len]));
        }
        k -= (PARAM_N / step);
        __syncwarp();
    }
    /* Scale by N^{-1} * MONT */
    for(int i = lane; i < PARAM_N; i += 32)
        r[i] = montgomery_reduce((coeff2_t)INTT_F * r[i]);
    __syncwarp();

#elif ALGORITHM == ALGO_AIGIS
    unsigned int ki = 0;
    #pragma unroll
    for(unsigned int len = 1; len <= 128; len <<= 1)
    {
        unsigned int step       = len << 1;
        unsigned int num_blocks = PARAM_N / step;
        for(int b = lane; b < PARAM_N / 2; b += 32)
        {
            unsigned int blk  = b / len;
            unsigned int pos  = b % len;
            unsigned int j    = blk * step + pos;
            coeff_t      zeta = ntt_zetas_inv[ki + blk];
            coeff_t      t    = r[j];
            r[j]              = t + r[j + len];
            r[j + len]        = montgomery_reduce((coeff2_t)zeta * (t - r[j + len]));
        }
        ki += num_blocks;
        __syncwarp();
    }
    /* Scale: only first N/2 coefficients (Aigis: last-stage twiddle bakes N^{-1} for upper half) */
    for(int i = lane; i < PARAM_N / 2; i += 32)
        r[i] = montgomery_reduce((coeff2_t)INTT_F * r[i]);
    __syncwarp();
#endif
}

#endif /* BATCH_NTT_HPP */
