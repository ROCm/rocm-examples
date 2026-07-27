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

#ifndef BATCH_KEM_HPP
#define BATCH_KEM_HPP

#include "batch_ntt.hpp"
#include "batch_ops.hpp"
#include "cbd.hpp"
#include "fips202.hpp"
#include "kem.hpp"
#include "ntt.hpp"
#include "params.h"
#include "poly.hpp"
#include "polyvec.hpp"
#include "reduce.hpp"
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>

struct BatchKemBuffers
{

    int16_t* d_mat;
    int16_t* d_skpv;
    int16_t* d_pkpv;
    int16_t* d_e;

    uint8_t* d_pk_bytes; /* B * PARAM_PUBLICKEYBYTES */
    uint8_t* d_sk_bytes; /* B * PARAM_SECRETKEYBYTES */
    uint8_t* d_ct_bytes; /* B * PARAM_CIPHERTEXTBYTES */
    uint8_t* d_ss_bytes; /* B * PARAM_SSBYTES */

    uint8_t* d_coins_kg;
    uint8_t* d_coins_enc;

    uint8_t* d_publicseed_kg;
    uint8_t* d_noiseseed_kg;

    int max_batch;
};

static inline void batch_kem_alloc(BatchKemBuffers* buf, int max_batch)
{
    buf->max_batch = max_batch;
    hipMalloc(&buf->d_mat, (size_t)PARAM_K * PARAM_K * max_batch * PARAM_N * sizeof(int16_t));
    hipMalloc(&buf->d_skpv, (size_t)PARAM_K * max_batch * PARAM_N * sizeof(int16_t));
    hipMalloc(&buf->d_pkpv, (size_t)PARAM_K * max_batch * PARAM_N * sizeof(int16_t));
    hipMalloc(&buf->d_e, (size_t)PARAM_K * max_batch * PARAM_N * sizeof(int16_t));
    hipMalloc(&buf->d_pk_bytes, (size_t)max_batch * PARAM_PUBLICKEYBYTES);
    hipMalloc(&buf->d_sk_bytes, (size_t)max_batch * PARAM_SECRETKEYBYTES);
    hipMalloc(&buf->d_ct_bytes, (size_t)max_batch * PARAM_CIPHERTEXTBYTES);
    hipMalloc(&buf->d_ss_bytes, (size_t)max_batch * PARAM_SSBYTES);
    hipMalloc(&buf->d_coins_kg, (size_t)max_batch * 2 * PARAM_SYMBYTES);
    hipMalloc(&buf->d_coins_enc, (size_t)max_batch * PARAM_SYMBYTES);
    hipMalloc(&buf->d_publicseed_kg, (size_t)max_batch * PARAM_SYMBYTES);
    hipMalloc(&buf->d_noiseseed_kg, (size_t)max_batch * PARAM_SYMBYTES);
}

static inline void batch_kem_free(BatchKemBuffers* buf)
{
    hipFree(buf->d_mat);
    hipFree(buf->d_skpv);
    hipFree(buf->d_pkpv);
    hipFree(buf->d_e);
    hipFree(buf->d_pk_bytes);
    hipFree(buf->d_sk_bytes);
    hipFree(buf->d_ct_bytes);
    hipFree(buf->d_ss_bytes);
    hipFree(buf->d_coins_kg);
    hipFree(buf->d_coins_enc);
    hipFree(buf->d_publicseed_kg);
    hipFree(buf->d_noiseseed_kg);
}

#ifndef WP_KG_WARP_SIZE
    #define WP_KG_WARP_SIZE 32
#endif

#ifndef WP_KG_WARPS_BLOCK
    #define WP_KG_WARPS_BLOCK 4
#endif

#define WP_KG_TPB (WP_KG_WARP_SIZE * WP_KG_WARPS_BLOCK)

#ifndef KEM_SPLIT_KEYGEN_SAMPLE
    #define KEM_SPLIT_KEYGEN_SAMPLE 0
#endif

#ifndef KEM_SERIAL_TPB
    #define KEM_SERIAL_TPB 64
#endif

#ifndef KEM_KEYGEN_TPB
    #define KEM_KEYGEN_TPB KEM_SERIAL_TPB
#endif

#ifndef KEM_ENCAPS_TPB
    #define KEM_ENCAPS_TPB KEM_SERIAL_TPB
#endif

#ifndef KEM_DECAPS_TPB
    #define KEM_DECAPS_TPB KEM_SERIAL_TPB
#endif

__global__ void
    batch_keygen_warp_sample_kernel(int16_t* __restrict__ d_mat, /* K*K * B * N */
                                    int16_t* __restrict__ d_skpv, /* K   * B * N */
                                    int16_t* __restrict__ d_e, /* K   * B * N */
                                    uint8_t* __restrict__ d_publicseed,
                                    const uint8_t* __restrict__ d_coins, /* B * 2*SYMBYTES */
                                    int batch_count)
{
    int inst = blockIdx.x * WP_KG_WARPS_BLOCK + (threadIdx.x / WP_KG_WARP_SIZE);
    int lane = threadIdx.x & (WP_KG_WARP_SIZE - 1);

    if(inst >= batch_count)
        return;

    __shared__ uint8_t ws_pub[WP_KG_WARPS_BLOCK][PARAM_SYMBYTES];
    __shared__ uint8_t ws_noise[WP_KG_WARPS_BLOCK][PARAM_SYMBYTES];

    int      warp_id    = threadIdx.x / WP_KG_WARP_SIZE;
    uint8_t* publicseed = ws_pub[warp_id];
    uint8_t* noiseseed  = ws_noise[warp_id];

    if(lane == 0)
    {

        uint8_t buf[2 * PARAM_SYMBYTES];
        sha3_512(buf, d_coins + inst * 2 * PARAM_SYMBYTES, PARAM_SYMBYTES);
        for(int i = 0; i < PARAM_SYMBYTES; i++)
        {
            publicseed[i]                                   = buf[i];
            d_publicseed[(size_t)inst * PARAM_SYMBYTES + i] = buf[i];
        }
        for(int i = 0; i < PARAM_SYMBYTES; i++)
            noiseseed[i] = buf[PARAM_SYMBYTES + i];
    }
    __syncwarp();

    int total_mat_polys = PARAM_K * PARAM_K;
    for(int p = lane; p < total_mat_polys; p += WP_KG_WARP_SIZE)
    {
        int row = p / PARAM_K;
        int col = p % PARAM_K;

        int16_t* dst = d_mat + ((size_t)(row * PARAM_K + col) * batch_count + inst) * PARAM_N;

        uint8_t extseed[PARAM_SYMBYTES + 2];
        for(int i = 0; i < PARAM_SYMBYTES; i++)
            extseed[i] = publicseed[i];

#if ALGORITHM == ALGO_KYBER
        extseed[PARAM_SYMBYTES]     = (uint8_t)col; /* j */
        extseed[PARAM_SYMBYTES + 1] = (uint8_t)row; /* i */
#elif ALGORITHM == ALGO_AIGIS_ENC
        extseed[PARAM_SYMBYTES]     = (uint8_t)row; /* i */
        extseed[PARAM_SYMBYTES + 1] = (uint8_t)col; /* j */
#endif

#if KEM_DIRECT_REJ_UNIFORM
        rej_uniform_xof(dst, publicseed, extseed[PARAM_SYMBYTES], extseed[PARAM_SYMBYTES + 1]);
#else
        keccak_state state;
        shake128_absorb_once(&state, extseed, PARAM_SYMBYTES + 2);

        unsigned int ctr = 0;
        uint8_t      buf[PARAM_GEN_MATRIX_NBLOCKS * PARAM_XOF_BLOCKBYTES];
        while(ctr < PARAM_N)
        {
            shake128_squeezeblocks(buf, PARAM_GEN_MATRIX_NBLOCKS, &state);
            ctr += rej_uniform(dst + ctr,
                               PARAM_N - ctr,
                               buf,
                               PARAM_GEN_MATRIX_NBLOCKS * PARAM_XOF_BLOCKBYTES);
        }
#endif
    }

    for(int i = lane; i < PARAM_K; i += WP_KG_WARP_SIZE)
    {
        int16_t* dst_s = d_skpv + ((size_t)i * batch_count + inst) * PARAM_N;
        poly_getnoise_s(dst_s, noiseseed, (uint8_t)i);
    }
    for(int i = lane; i < PARAM_K; i += WP_KG_WARP_SIZE)
    {
        int16_t* dst_e = d_e + ((size_t)i * batch_count + inst) * PARAM_N;
        poly_getnoise_e_kg(dst_e, noiseseed, (uint8_t)(PARAM_K + i));
    }
}

__global__ void batch_keygen_seed_expand_kernel(uint8_t* __restrict__ d_publicseed,
                                                uint8_t* __restrict__ d_noiseseed,
                                                const uint8_t* __restrict__ d_coins,
                                                int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if(inst >= batch_count)
        return;

    uint8_t buf[2 * PARAM_SYMBYTES];
    sha3_512(buf, d_coins + (size_t)inst * 2 * PARAM_SYMBYTES, PARAM_SYMBYTES);
    for(int i = 0; i < PARAM_SYMBYTES; i++)
    {
        d_publicseed[(size_t)inst * PARAM_SYMBYTES + i] = buf[i];
        d_noiseseed[(size_t)inst * PARAM_SYMBYTES + i]  = buf[PARAM_SYMBYTES + i];
    }
}

__global__ void batch_keygen_mat_sample_kernel(int16_t* __restrict__ d_mat,
                                               const uint8_t* __restrict__ d_publicseed,
                                               int batch_count)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * PARAM_K * PARAM_K;
    if(idx >= total)
        return;

    int inst = idx % batch_count;
    int p    = idx / batch_count;
    int row  = p / PARAM_K;
    int col  = p % PARAM_K;

#if ALGORITHM == ALGO_KYBER
    uint8_t x = (uint8_t)col;
    uint8_t y = (uint8_t)row;
#elif ALGORITHM == ALGO_AIGIS_ENC
    uint8_t x = (uint8_t)row;
    uint8_t y = (uint8_t)col;
#endif

    int16_t*       dst  = d_mat + ((size_t)(row * PARAM_K + col) * batch_count + inst) * PARAM_N;
    const uint8_t* seed = d_publicseed + (size_t)inst * PARAM_SYMBYTES;
    rej_uniform_xof(dst, seed, x, y);
}

__global__ void batch_keygen_noise_sample_kernel(int16_t* __restrict__ d_skpv,
                                                 int16_t* __restrict__ d_e,
                                                 const uint8_t* __restrict__ d_noiseseed,
                                                 int batch_count)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * PARAM_K * 2;
    if(idx >= total)
        return;

    int            inst = idx % batch_count;
    int            q    = idx / batch_count;
    int            poly = q % PARAM_K;
    const uint8_t* seed = d_noiseseed + (size_t)inst * PARAM_SYMBYTES;

    if(q < PARAM_K)
    {
        int16_t* dst = d_skpv + ((size_t)poly * batch_count + inst) * PARAM_N;
        poly_getnoise_s(dst, seed, (uint8_t)poly);
    }
    else
    {
        int16_t* dst = d_e + ((size_t)poly * batch_count + inst) * PARAM_N;
        poly_getnoise_e_kg(dst, seed, (uint8_t)(PARAM_K + poly));
    }
}

__global__ void batch_pack_keypair_kernel(uint8_t* __restrict__ d_pk_bytes,
                                          uint8_t* __restrict__ d_sk_bytes,
                                          const int16_t* __restrict__ d_skpv,
                                          const int16_t* __restrict__ d_pkpv,
                                          const uint8_t* __restrict__ d_coins,
                                          int batch_count)
{
    int inst = blockIdx.x;
    if(inst >= batch_count)
        return;

    kem_polyvec skpv_local, pkpv_local;
    for(int i = 0; i < PARAM_K; i++)
        for(int c = 0; c < PARAM_N; c++)
        {
            skpv_local.vec[i].coeffs[c] = d_skpv[((size_t)i * batch_count + inst) * PARAM_N + c];
            pkpv_local.vec[i].coeffs[c] = d_pkpv[((size_t)i * batch_count + inst) * PARAM_N + c];
        }

    uint8_t seeds[2 * PARAM_SYMBYTES];
    sha3_512(seeds, d_coins + (size_t)inst * 2 * PARAM_SYMBYTES, PARAM_SYMBYTES);
    const uint8_t* publicseed = seeds;

    /* PK = pk_poly_compress(pkpv) || publicseed */
    uint8_t* pk = d_pk_bytes + (size_t)inst * PARAM_PUBLICKEYBYTES;
    pack_pk(pk, &pkpv_local, publicseed);

    /* SK = polyvec_tobytes(skpv) || pk || H(pk) || z */
    uint8_t* sk = d_sk_bytes + (size_t)inst * PARAM_SECRETKEYBYTES;
    pack_sk(sk, &skpv_local);

    /* sk[indcpa_sk_bytes:] = pk */
    for(int i = 0; i < (int)PARAM_PUBLICKEYBYTES; i++)
        sk[PARAM_INDCPA_SECRETKEYBYTES + i] = pk[i];

    /* H(pk) */
    sha3_256(sk + PARAM_INDCPA_SECRETKEYBYTES + PARAM_PUBLICKEYBYTES, pk, PARAM_PUBLICKEYBYTES);

    const uint8_t* z_src = d_coins + (size_t)inst * 2 * PARAM_SYMBYTES + PARAM_SYMBYTES;
    uint8_t*       z_dst = sk + PARAM_INDCPA_SECRETKEYBYTES + PARAM_PUBLICKEYBYTES + PARAM_SYMBYTES;
    for(int i = 0; i < PARAM_SYMBYTES; i++)
        z_dst[i] = z_src[i];
}

#ifndef KEM_PACK_TPB
    #define KEM_PACK_TPB 128
#endif

__global__ void batch_pack_sk_polyvec_kernel(uint8_t* __restrict__ d_sk_bytes,
                                             const int16_t* __restrict__ d_skpv,
                                             int batch_count)
{
    int inst = blockIdx.x;
    int poly = blockIdx.y;
    int tid  = threadIdx.x;
    if(inst >= batch_count || poly >= PARAM_K)
        return;

    const int16_t* src = d_skpv + ((size_t)poly * batch_count + inst) * PARAM_N;
    uint8_t*       out
        = d_sk_bytes + (size_t)inst * PARAM_SECRETKEYBYTES + (size_t)poly * PARAM_POLYBYTES;

#if ALGORITHM == ALGO_KYBER
    for(int i = tid; i < PARAM_N / 2; i += blockDim.x)
    {
        int16_t t0     = caddq(src[2 * i]);
        int16_t t1     = caddq(src[2 * i + 1]);
        out[3 * i + 0] = (uint8_t)t0;
        out[3 * i + 1] = (uint8_t)((t0 >> 8) | (t1 << 4));
        out[3 * i + 2] = (uint8_t)(t1 >> 4);
    }
#elif ALGORITHM == ALGO_AIGIS_ENC
    for(int i = tid; i < PARAM_N / 8; i += blockDim.x)
    {
        int16_t t0       = caddq(src[8 * i + 0]);
        int16_t t1       = caddq(src[8 * i + 1]);
        int16_t t2       = caddq(src[8 * i + 2]);
        int16_t t3       = caddq(src[8 * i + 3]);
        int16_t t4       = caddq(src[8 * i + 4]);
        int16_t t5       = caddq(src[8 * i + 5]);
        int16_t t6       = caddq(src[8 * i + 6]);
        int16_t t7       = caddq(src[8 * i + 7]);
        out[13 * i + 0]  = (uint8_t)t0;
        out[13 * i + 1]  = (uint8_t)((t0 >> 8) | (t1 << 5));
        out[13 * i + 2]  = (uint8_t)(t1 >> 3);
        out[13 * i + 3]  = (uint8_t)((t1 >> 11) | (t2 << 2));
        out[13 * i + 4]  = (uint8_t)((t2 >> 6) | (t3 << 7));
        out[13 * i + 5]  = (uint8_t)(t3 >> 1);
        out[13 * i + 6]  = (uint8_t)((t3 >> 9) | (t4 << 4));
        out[13 * i + 7]  = (uint8_t)(t4 >> 4);
        out[13 * i + 8]  = (uint8_t)((t4 >> 12) | (t5 << 1));
        out[13 * i + 9]  = (uint8_t)((t5 >> 7) | (t6 << 6));
        out[13 * i + 10] = (uint8_t)(t6 >> 2);
        out[13 * i + 11] = (uint8_t)((t6 >> 10) | (t7 << 3));
        out[13 * i + 12] = (uint8_t)(t7 >> 5);
    }
#endif
}

__global__ void batch_pack_pk_polyvec_kernel(uint8_t* __restrict__ d_pk_bytes,
                                             const int16_t* __restrict__ d_pkpv,
                                             int batch_count)
{
    int inst = blockIdx.x;
    int poly = blockIdx.y;
    int tid  = threadIdx.x;
    if(inst >= batch_count || poly >= PARAM_K)
        return;

    const int16_t* src = d_pkpv + ((size_t)poly * batch_count + inst) * PARAM_N;
    uint8_t*       out = d_pk_bytes + (size_t)inst * PARAM_PUBLICKEYBYTES
                   + (size_t)poly * (PARAM_BITS_PK * PARAM_N / 8);

#if ALGORITHM == ALGO_KYBER
    for(int i = tid; i < PARAM_N / 2; i += blockDim.x)
    {
        int16_t t0     = caddq(src[2 * i]);
        int16_t t1     = caddq(src[2 * i + 1]);
        out[3 * i + 0] = (uint8_t)t0;
        out[3 * i + 1] = (uint8_t)((t0 >> 8) | (t1 << 4));
        out[3 * i + 2] = (uint8_t)(t1 >> 4);
    }
#elif PARAM_BITS_PK == 9
    for(int i = tid; i < PARAM_N / 8; i += blockDim.x)
    {
        uint16_t c0
            = (uint16_t)((((int32_t)caddq(src[8 * i + 0]) << 9) + PARAM_Q / 2) / PARAM_Q) & 0x1FF;
        uint16_t c1
            = (uint16_t)((((int32_t)caddq(src[8 * i + 1]) << 9) + PARAM_Q / 2) / PARAM_Q) & 0x1FF;
        uint16_t c2
            = (uint16_t)((((int32_t)caddq(src[8 * i + 2]) << 9) + PARAM_Q / 2) / PARAM_Q) & 0x1FF;
        uint16_t c3
            = (uint16_t)((((int32_t)caddq(src[8 * i + 3]) << 9) + PARAM_Q / 2) / PARAM_Q) & 0x1FF;
        uint16_t c4
            = (uint16_t)((((int32_t)caddq(src[8 * i + 4]) << 9) + PARAM_Q / 2) / PARAM_Q) & 0x1FF;
        uint16_t c5
            = (uint16_t)((((int32_t)caddq(src[8 * i + 5]) << 9) + PARAM_Q / 2) / PARAM_Q) & 0x1FF;
        uint16_t c6
            = (uint16_t)((((int32_t)caddq(src[8 * i + 6]) << 9) + PARAM_Q / 2) / PARAM_Q) & 0x1FF;
        uint16_t c7
            = (uint16_t)((((int32_t)caddq(src[8 * i + 7]) << 9) + PARAM_Q / 2) / PARAM_Q) & 0x1FF;
        out[9 * i + 0] = (uint8_t)c0;
        out[9 * i + 1] = (uint8_t)((c0 >> 8) | (c1 << 1));
        out[9 * i + 2] = (uint8_t)((c1 >> 7) | (c2 << 2));
        out[9 * i + 3] = (uint8_t)((c2 >> 6) | (c3 << 3));
        out[9 * i + 4] = (uint8_t)((c3 >> 5) | (c4 << 4));
        out[9 * i + 5] = (uint8_t)((c4 >> 4) | (c5 << 5));
        out[9 * i + 6] = (uint8_t)((c5 >> 3) | (c6 << 6));
        out[9 * i + 7] = (uint8_t)((c6 >> 2) | (c7 << 7));
        out[9 * i + 8] = (uint8_t)(c7 >> 1);
    }
#elif PARAM_BITS_PK == 10
    for(int i = tid; i < PARAM_N / 4; i += blockDim.x)
    {
        uint16_t c0
            = (uint16_t)((((int32_t)caddq(src[4 * i + 0]) << 10) + PARAM_Q / 2) / PARAM_Q) & 0x3FF;
        uint16_t c1
            = (uint16_t)((((int32_t)caddq(src[4 * i + 1]) << 10) + PARAM_Q / 2) / PARAM_Q) & 0x3FF;
        uint16_t c2
            = (uint16_t)((((int32_t)caddq(src[4 * i + 2]) << 10) + PARAM_Q / 2) / PARAM_Q) & 0x3FF;
        uint16_t c3
            = (uint16_t)((((int32_t)caddq(src[4 * i + 3]) << 10) + PARAM_Q / 2) / PARAM_Q) & 0x3FF;
        out[5 * i + 0] = (uint8_t)c0;
        out[5 * i + 1] = (uint8_t)((c0 >> 8) | (c1 << 2));
        out[5 * i + 2] = (uint8_t)((c1 >> 6) | (c2 << 4));
        out[5 * i + 3] = (uint8_t)((c2 >> 4) | (c3 << 6));
        out[5 * i + 4] = (uint8_t)(c3 >> 2);
    }
#elif PARAM_BITS_PK == 11
    for(int i = tid; i < PARAM_N / 8; i += blockDim.x)
    {
        uint16_t c0
            = (uint16_t)((((int32_t)caddq(src[8 * i + 0]) << 11) + PARAM_Q / 2) / PARAM_Q) & 0x7FF;
        uint16_t c1
            = (uint16_t)((((int32_t)caddq(src[8 * i + 1]) << 11) + PARAM_Q / 2) / PARAM_Q) & 0x7FF;
        uint16_t c2
            = (uint16_t)((((int32_t)caddq(src[8 * i + 2]) << 11) + PARAM_Q / 2) / PARAM_Q) & 0x7FF;
        uint16_t c3
            = (uint16_t)((((int32_t)caddq(src[8 * i + 3]) << 11) + PARAM_Q / 2) / PARAM_Q) & 0x7FF;
        uint16_t c4
            = (uint16_t)((((int32_t)caddq(src[8 * i + 4]) << 11) + PARAM_Q / 2) / PARAM_Q) & 0x7FF;
        uint16_t c5
            = (uint16_t)((((int32_t)caddq(src[8 * i + 5]) << 11) + PARAM_Q / 2) / PARAM_Q) & 0x7FF;
        uint16_t c6
            = (uint16_t)((((int32_t)caddq(src[8 * i + 6]) << 11) + PARAM_Q / 2) / PARAM_Q) & 0x7FF;
        uint16_t c7
            = (uint16_t)((((int32_t)caddq(src[8 * i + 7]) << 11) + PARAM_Q / 2) / PARAM_Q) & 0x7FF;
        out[11 * i + 0]  = (uint8_t)c0;
        out[11 * i + 1]  = (uint8_t)((c0 >> 8) | (c1 << 3));
        out[11 * i + 2]  = (uint8_t)((c1 >> 5) | (c2 << 6));
        out[11 * i + 3]  = (uint8_t)(c2 >> 2);
        out[11 * i + 4]  = (uint8_t)((c2 >> 10) | (c3 << 1));
        out[11 * i + 5]  = (uint8_t)((c3 >> 7) | (c4 << 4));
        out[11 * i + 6]  = (uint8_t)((c4 >> 4) | (c5 << 7));
        out[11 * i + 7]  = (uint8_t)(c5 >> 1);
        out[11 * i + 8]  = (uint8_t)((c5 >> 9) | (c6 << 2));
        out[11 * i + 9]  = (uint8_t)((c6 >> 6) | (c7 << 5));
        out[11 * i + 10] = (uint8_t)(c7 >> 3);
    }
#endif
}

__global__ void batch_pack_keypair_finalize_kernel(uint8_t* __restrict__ d_pk_bytes,
                                                   uint8_t* __restrict__ d_sk_bytes,
                                                   const uint8_t* __restrict__ d_publicseed,
                                                   const uint8_t* __restrict__ d_coins,
                                                   int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if(inst >= batch_count)
        return;

    uint8_t*       pk  = d_pk_bytes + (size_t)inst * PARAM_PUBLICKEYBYTES;
    uint8_t*       sk  = d_sk_bytes + (size_t)inst * PARAM_SECRETKEYBYTES;
    const uint8_t* rho = d_publicseed + (size_t)inst * PARAM_SYMBYTES;

    for(int i = 0; i < PARAM_SYMBYTES; i++)
        pk[PARAM_PK_POLYVEC_BYTES + i] = rho[i];

    for(int i = 0; i < (int)PARAM_PUBLICKEYBYTES; i++)
        sk[PARAM_INDCPA_SECRETKEYBYTES + i] = pk[i];

    sha3_256(sk + PARAM_INDCPA_SECRETKEYBYTES + PARAM_PUBLICKEYBYTES, pk, PARAM_PUBLICKEYBYTES);

    const uint8_t* z_src = d_coins + (size_t)inst * 2 * PARAM_SYMBYTES + PARAM_SYMBYTES;
    uint8_t*       z_dst = sk + PARAM_INDCPA_SECRETKEYBYTES + PARAM_PUBLICKEYBYTES + PARAM_SYMBYTES;
    for(int i = 0; i < PARAM_SYMBYTES; i++)
        z_dst[i] = z_src[i];
}

#ifndef KEM_KEYPAIR_LAUNCH_BOUNDS
    #define KEM_KEYPAIR_LAUNCH_BOUNDS 1
#endif

#ifndef KEM_ENCAPS_LAUNCH_BOUNDS
    #if ALGORITHM == ALGO_AIGIS_ENC
        #define KEM_ENCAPS_LAUNCH_BOUNDS 1
    #else
        #define KEM_ENCAPS_LAUNCH_BOUNDS 0
    #endif
#endif

#ifndef KEM_DECAPS_LAUNCH_BOUNDS
    #if ALGORITHM == ALGO_AIGIS_ENC
        #define KEM_DECAPS_LAUNCH_BOUNDS 1
    #else
        #define KEM_DECAPS_LAUNCH_BOUNDS 0
    #endif
#endif

#if KEM_KEYPAIR_LAUNCH_BOUNDS
    #define KEM_KEYPAIR_KERNEL_BOUNDS __launch_bounds__(KEM_KEYGEN_TPB, 1)
#else
    #define KEM_KEYPAIR_KERNEL_BOUNDS
#endif

#if KEM_ENCAPS_LAUNCH_BOUNDS
    #define KEM_ENCAPS_KERNEL_BOUNDS __launch_bounds__(KEM_ENCAPS_TPB, 1)
#else
    #define KEM_ENCAPS_KERNEL_BOUNDS
#endif

#if KEM_DECAPS_LAUNCH_BOUNDS
    #define KEM_DECAPS_KERNEL_BOUNDS __launch_bounds__(KEM_DECAPS_TPB, 1)
#else
    #define KEM_DECAPS_KERNEL_BOUNDS
#endif

__global__ KEM_KEYPAIR_KERNEL_BOUNDS void
    batch_kem_keypair_serial_kernel(uint8_t* __restrict__ d_pk,
                                    uint8_t* __restrict__ d_sk,
                                    const uint8_t* __restrict__ d_coins, /* B * 2*SYMBYTES */
                                    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if(inst >= batch_count)
        return;

    kem_keypair(d_pk + (size_t)inst * PARAM_PUBLICKEYBYTES,
                d_sk + (size_t)inst * PARAM_SECRETKEYBYTES,
                d_coins + (size_t)inst * 2 * PARAM_SYMBYTES);
}

__global__ KEM_ENCAPS_KERNEL_BOUNDS void
    batch_kem_encaps_serial_kernel(uint8_t* __restrict__ d_ct,
                                   uint8_t* __restrict__ d_ss,
                                   const uint8_t* __restrict__ d_pk,
                                   const uint8_t* __restrict__ d_coins, /* B * SYMBYTES */
                                   int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if(inst >= batch_count)
        return;

    kem_encaps(d_ct + (size_t)inst * PARAM_CIPHERTEXTBYTES,
               d_ss + (size_t)inst * PARAM_SSBYTES,
               d_pk + (size_t)inst * PARAM_PUBLICKEYBYTES,
               d_coins + (size_t)inst * PARAM_SYMBYTES);
}

__global__ KEM_DECAPS_KERNEL_BOUNDS void
    batch_kem_decaps_serial_kernel(uint8_t* __restrict__ d_ss,
                                   const uint8_t* __restrict__ d_ct,
                                   const uint8_t* __restrict__ d_sk,
                                   int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if(inst >= batch_count)
        return;

    kem_decaps(d_ss + (size_t)inst * PARAM_SSBYTES,
               d_ct + (size_t)inst * PARAM_CIPHERTEXTBYTES,
               d_sk + (size_t)inst * PARAM_SECRETKEYBYTES);
}

static inline hipError_t batch_keygen_pipelined(uint8_t*         d_pk_out,
                                                uint8_t*         d_sk_out,
                                                BatchKemBuffers* buf,
                                                int              batch_count,
                                                hipStream_t      stream = 0)
{
    hipError_t err;

    int blocks = (batch_count + WP_KG_WARPS_BLOCK - 1) / WP_KG_WARPS_BLOCK;
#if KEM_SPLIT_KEYGEN_SAMPLE
    batch_keygen_seed_expand_kernel<<<ceil_div(batch_count, KEM_SERIAL_TPB),
                                      KEM_SERIAL_TPB,
                                      0,
                                      stream>>>(buf->d_publicseed_kg,
                                                buf->d_noiseseed_kg,
                                                buf->d_coins_kg,
                                                batch_count);
    batch_keygen_mat_sample_kernel<<<ceil_div(batch_count * PARAM_K * PARAM_K, KEM_SERIAL_TPB),
                                     KEM_SERIAL_TPB,
                                     0,
                                     stream>>>(buf->d_mat, buf->d_publicseed_kg, batch_count);
    batch_keygen_noise_sample_kernel<<<ceil_div(batch_count * PARAM_K * 2, KEM_SERIAL_TPB),
                                       KEM_SERIAL_TPB,
                                       0,
                                       stream>>>(buf->d_skpv,
                                                 buf->d_e,
                                                 buf->d_noiseseed_kg,
                                                 batch_count);
#else
    batch_keygen_warp_sample_kernel<<<blocks, WP_KG_TPB, 0, stream>>>(buf->d_mat,
                                                                      buf->d_skpv,
                                                                      buf->d_e,
                                                                      buf->d_publicseed_kg,
                                                                      buf->d_coins_kg,
                                                                      batch_count);
#endif

    for(int i = 0; i < PARAM_K; i++)
    {
        int16_t* ptr = buf->d_skpv + (size_t)i * batch_count * PARAM_N;
        batch_ntt_kernel<<<batch_count, 128, 0, stream>>>(ptr, batch_count);
    }

    /* Step 2b: caddq(s) */
    for(int i = 0; i < PARAM_K; i++)
    {
        int16_t* ptr = buf->d_skpv + (size_t)i * batch_count * PARAM_N;
        launch_batch_caddq(ptr, batch_count, stream);
    }

    launch_batch_matvec(buf->d_pkpv, buf->d_mat, buf->d_skpv, batch_count, stream);

    /* Step 4: INVNTT(pkpv) */
    for(int i = 0; i < PARAM_K; i++)
    {
        int16_t* ptr = buf->d_pkpv + (size_t)i * batch_count * PARAM_N;
        batch_invntt_kernel<<<batch_count, 128, 0, stream>>>(ptr, batch_count);
    }

    /* pkpv += e */
    for(int i = 0; i < PARAM_K; i++)
    {
        launch_batch_add(buf->d_pkpv + (size_t)i * batch_count * PARAM_N,
                         buf->d_pkpv + (size_t)i * batch_count * PARAM_N,
                         buf->d_e + (size_t)i * batch_count * PARAM_N,
                         batch_count,
                         stream);
    }

    /* caddq(pkpv) */
    for(int i = 0; i < PARAM_K; i++)
    {
        launch_batch_caddq(buf->d_pkpv + (size_t)i * batch_count * PARAM_N, batch_count, stream);
    }

    dim3 pack_grid(batch_count, PARAM_K);
    batch_pack_sk_polyvec_kernel<<<pack_grid, KEM_PACK_TPB, 0, stream>>>(d_sk_out,
                                                                         buf->d_skpv,
                                                                         batch_count);
    batch_pack_pk_polyvec_kernel<<<pack_grid, KEM_PACK_TPB, 0, stream>>>(d_pk_out,
                                                                         buf->d_pkpv,
                                                                         batch_count);
    batch_pack_keypair_finalize_kernel<<<ceil_div(batch_count, KEM_SERIAL_TPB),
                                         KEM_SERIAL_TPB,
                                         0,
                                         stream>>>(d_pk_out,
                                                   d_sk_out,
                                                   buf->d_publicseed_kg,
                                                   buf->d_coins_kg,
                                                   batch_count);

    err = hipGetLastError();
    return err;
}

static inline hipError_t batch_keygen_pipelined_profile(uint8_t*         d_pk_out,
                                                        uint8_t*         d_sk_out,
                                                        BatchKemBuffers* buf,
                                                        int              batch_count,
                                                        hipStream_t      stream = 0)
{
    hipEvent_t ev0, ev1, ev2, ev3, ev4, ev5, ev6;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);
    hipEventCreate(&ev2);
    hipEventCreate(&ev3);
    hipEventCreate(&ev4);
    hipEventCreate(&ev5);
    hipEventCreate(&ev6);

    hipEventRecord(ev0, stream);
    int blocks = (batch_count + WP_KG_WARPS_BLOCK - 1) / WP_KG_WARPS_BLOCK;
#if KEM_SPLIT_KEYGEN_SAMPLE
    batch_keygen_seed_expand_kernel<<<ceil_div(batch_count, KEM_SERIAL_TPB),
                                      KEM_SERIAL_TPB,
                                      0,
                                      stream>>>(buf->d_publicseed_kg,
                                                buf->d_noiseseed_kg,
                                                buf->d_coins_kg,
                                                batch_count);
    batch_keygen_mat_sample_kernel<<<ceil_div(batch_count * PARAM_K * PARAM_K, KEM_SERIAL_TPB),
                                     KEM_SERIAL_TPB,
                                     0,
                                     stream>>>(buf->d_mat, buf->d_publicseed_kg, batch_count);
    batch_keygen_noise_sample_kernel<<<ceil_div(batch_count * PARAM_K * 2, KEM_SERIAL_TPB),
                                       KEM_SERIAL_TPB,
                                       0,
                                       stream>>>(buf->d_skpv,
                                                 buf->d_e,
                                                 buf->d_noiseseed_kg,
                                                 batch_count);
#else
    batch_keygen_warp_sample_kernel<<<blocks, WP_KG_TPB, 0, stream>>>(buf->d_mat,
                                                                      buf->d_skpv,
                                                                      buf->d_e,
                                                                      buf->d_publicseed_kg,
                                                                      buf->d_coins_kg,
                                                                      batch_count);
#endif
    hipEventRecord(ev1, stream);

    for(int i = 0; i < PARAM_K; i++)
    {
        int16_t* ptr = buf->d_skpv + (size_t)i * batch_count * PARAM_N;
        batch_ntt_kernel<<<batch_count, 128, 0, stream>>>(ptr, batch_count);
    }
    for(int i = 0; i < PARAM_K; i++)
    {
        int16_t* ptr = buf->d_skpv + (size_t)i * batch_count * PARAM_N;
        launch_batch_caddq(ptr, batch_count, stream);
    }
    hipEventRecord(ev2, stream);

    launch_batch_matvec(buf->d_pkpv, buf->d_mat, buf->d_skpv, batch_count, stream);
    hipEventRecord(ev3, stream);

    for(int i = 0; i < PARAM_K; i++)
    {
        int16_t* ptr = buf->d_pkpv + (size_t)i * batch_count * PARAM_N;
        batch_invntt_kernel<<<batch_count, 128, 0, stream>>>(ptr, batch_count);
    }
    hipEventRecord(ev4, stream);

    for(int i = 0; i < PARAM_K; i++)
    {
        launch_batch_add(buf->d_pkpv + (size_t)i * batch_count * PARAM_N,
                         buf->d_pkpv + (size_t)i * batch_count * PARAM_N,
                         buf->d_e + (size_t)i * batch_count * PARAM_N,
                         batch_count,
                         stream);
    }
    for(int i = 0; i < PARAM_K; i++)
        launch_batch_caddq(buf->d_pkpv + (size_t)i * batch_count * PARAM_N, batch_count, stream);
    hipEventRecord(ev5, stream);

    dim3 pack_grid(batch_count, PARAM_K);
    batch_pack_sk_polyvec_kernel<<<pack_grid, KEM_PACK_TPB, 0, stream>>>(d_sk_out,
                                                                         buf->d_skpv,
                                                                         batch_count);
    batch_pack_pk_polyvec_kernel<<<pack_grid, KEM_PACK_TPB, 0, stream>>>(d_pk_out,
                                                                         buf->d_pkpv,
                                                                         batch_count);
    batch_pack_keypair_finalize_kernel<<<ceil_div(batch_count, KEM_SERIAL_TPB),
                                         KEM_SERIAL_TPB,
                                         0,
                                         stream>>>(d_pk_out,
                                                   d_sk_out,
                                                   buf->d_publicseed_kg,
                                                   buf->d_coins_kg,
                                                   batch_count);
    hipEventRecord(ev6, stream);
    hipEventSynchronize(ev6);

    float sample_ms, ntt_ms, matvec_ms, invntt_ms, add_ms, pack_ms, total_ms;
    hipEventElapsedTime(&sample_ms, ev0, ev1);
    hipEventElapsedTime(&ntt_ms, ev1, ev2);
    hipEventElapsedTime(&matvec_ms, ev2, ev3);
    hipEventElapsedTime(&invntt_ms, ev3, ev4);
    hipEventElapsedTime(&add_ms, ev4, ev5);
    hipEventElapsedTime(&pack_ms, ev5, ev6);
    hipEventElapsedTime(&total_ms, ev0, ev6);
    printf("  Pipeline profile: sample=%.3f ntt=%.3f matvec=%.3f invntt=%.3f add=%.3f pack=%.3f "
           "total=%.3f ms\n",
           sample_ms,
           ntt_ms,
           matvec_ms,
           invntt_ms,
           add_ms,
           pack_ms,
           total_ms);

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    hipEventDestroy(ev2);
    hipEventDestroy(ev3);
    hipEventDestroy(ev4);
    hipEventDestroy(ev5);
    hipEventDestroy(ev6);
    return hipGetLastError();
}

static inline hipError_t batch_encaps_serial(uint8_t*         d_ct,
                                             uint8_t*         d_ss,
                                             const uint8_t*   d_pk,
                                             BatchKemBuffers* buf,
                                             int              batch_count,
                                             hipStream_t      stream = 0)
{
    int tpb    = KEM_ENCAPS_TPB;
    int blocks = (batch_count + tpb - 1) / tpb;
    batch_kem_encaps_serial_kernel<<<blocks, tpb, 0, stream>>>(d_ct,
                                                               d_ss,
                                                               d_pk,
                                                               buf->d_coins_enc,
                                                               batch_count);
    return hipGetLastError();
}

static inline hipError_t batch_decaps_serial(uint8_t*       d_ss,
                                             const uint8_t* d_ct,
                                             const uint8_t* d_sk,
                                             int            batch_count,
                                             hipStream_t    stream = 0)
{
    int tpb    = KEM_DECAPS_TPB;
    int blocks = (batch_count + tpb - 1) / tpb;
    batch_kem_decaps_serial_kernel<<<blocks, tpb, 0, stream>>>(d_ss, d_ct, d_sk, batch_count);
    return hipGetLastError();
}

#endif /* BATCH_KEM_HPP */
