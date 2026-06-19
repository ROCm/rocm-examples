/*
 * poly.cuh — 统一多项式运算
 *
 * 支持 Kyber (Q=3329, 12-bit) 和 Aigis-enc (Q=7681, 13-bit)
 * 差异通过 #if ALGORITHM 编译时分支处理
 *
 * 主要差异:
 *   frommsg/tomsg: Kyber 阈值 Q/2, Aigis 阈值 (Q+1)/2
 *   tobytes/frombytes: 12-bit (Kyber) vs 13-bit (Aigis)
 *   compress_c2/decompress_c2: 4-bit 或 5-bit (Kyber) / 3-bit,4-bit,5-bit (Aigis)
 */

#ifndef POLY_CUH
#define POLY_CUH

#include <stdint.h>
#include "params.h"
#include "reduce.cuh"
#include "ntt.cuh"
#include "cbd.cuh"

/* ================================================================
 *  基础多项式算术
 * ================================================================ */

static __device__ void poly_add(kem_poly *r, const kem_poly *a, const kem_poly *b)
{
    for (int i = 0; i < PARAM_N; i++) r->coeffs[i] = a->coeffs[i] + b->coeffs[i];
}

static __device__ void poly_sub(kem_poly *r, const kem_poly *a, const kem_poly *b)
{
    for (int i = 0; i < PARAM_N; i++) r->coeffs[i] = a->coeffs[i] - b->coeffs[i];
}

static __device__ void poly_reduce(kem_poly *r)
{
    for (int i = 0; i < PARAM_N; i++) r->coeffs[i] = barrett_reduce(r->coeffs[i]);
}

static __device__ void poly_caddq(kem_poly *r)
{
    for (int i = 0; i < PARAM_N; i++) r->coeffs[i] = caddq(r->coeffs[i]);
}

static __device__ void poly_caddq2(kem_poly *r)
{
    for (int i = 0; i < PARAM_N; i++) r->coeffs[i] = caddq2(r->coeffs[i]);
}

/* ================================================================
 *  消息编解码
 *  frommsg: {0,1}^256 → poly in [0, Q)
 *  tomsg:   poly → {0,1}^256
 * ================================================================ */

static __device__ void poly_frommsg(kem_poly *r, const uint8_t *msg)
{
    for (int i = 0; i < PARAM_N / 8; i++) {
        for (int j = 0; j < 8; j++) {
            int16_t mask = -((msg[i] >> j) & 1);  /* 0xFFFF if bit=1, else 0 */
#if ALGORITHM == ALGO_KYBER
            /* Kyber: bit=1 → (Q+1)/2 = 1665 */
            r->coeffs[8*i+j] = mask & (int16_t)((PARAM_Q + 1) / 2);
#elif ALGORITHM == ALGO_AIGIS_ENC
            /* Aigis: bit=1 → (Q+1)/2 = 3841 */
            r->coeffs[8*i+j] = mask & (int16_t)((PARAM_Q + 1) / 2);
#endif
        }
    }
}

static __device__ void poly_tomsg(uint8_t *msg, const kem_poly *r)
{
    for (int i = 0; i < PARAM_N / 8; i++) {
        msg[i] = 0;
        for (int j = 0; j < 8; j++) {
            /* 四舍五入到最近整数 mod 2: 若系数更接近 Q/2 则为 1 */
            int16_t t = r->coeffs[8*i+j];
            t = caddq(t);
            /* 放大到 2 个区间: (t * 2 + Q/2) / Q & 1 */
            /* Kyber 公式: ((t << 1) + Q/2) / Q & 1 */
#if ALGORITHM == ALGO_KYBER
            /* 阈值测试: 若 t > Q/4 且 t < 3Q/4 则为 1 */
            t = (int16_t)(((t << 1) + PARAM_Q / 2) / PARAM_Q);
#elif ALGORITHM == ALGO_AIGIS_ENC
            t = (int16_t)(((t << 1) + PARAM_Q / 2) / PARAM_Q);
#endif
            msg[i] |= (uint8_t)((t & 1) << j);
        }
    }
}

/* ================================================================
 *  序列化/反序列化 (全精度)
 *  Kyber:    12-bit per coeff → 384 bytes
 *  Aigis-enc:13-bit per coeff → 416 bytes
 * ================================================================ */

#if ALGORITHM == ALGO_KYBER

/* 12-bit → 384 bytes */
static __device__ __noinline__ void poly_tobytes(uint8_t *r, const kem_poly *a)
{
    for (unsigned int i = 0; i < PARAM_N / 2; i++) {
        int16_t t0 = caddq(a->coeffs[2*i]);
        int16_t t1 = caddq(a->coeffs[2*i+1]);
        r[3*i+0] = (uint8_t)(t0);
        r[3*i+1] = (uint8_t)((t0 >> 8) | (t1 << 4));
        r[3*i+2] = (uint8_t)(t1 >> 4);
    }
}

static __device__ __noinline__ void poly_frombytes(kem_poly *r, const uint8_t *a)
{
    for (unsigned int i = 0; i < PARAM_N / 2; i++) {
        r->coeffs[2*i]   = (int16_t)(((a[3*i+0])       | ((int16_t)a[3*i+1] << 8)) & 0xFFF);
        r->coeffs[2*i+1] = (int16_t)(((a[3*i+1] >> 4)  | ((int16_t)a[3*i+2] << 4)) & 0xFFF);
    }
}

#elif ALGORITHM == ALGO_AIGIS_ENC

/* 13-bit → 416 bytes (8 coeffs per 13 bytes) */
static __device__ __noinline__ void poly_tobytes(uint8_t *r, const kem_poly *a)
{
    for (unsigned int i = 0; i < PARAM_N / 8; i++) {
        int16_t t[8];
        for (int j = 0; j < 8; j++) t[j] = caddq(a->coeffs[8*i+j]);
        r[13*i+ 0] = (uint8_t)(t[0]);
        r[13*i+ 1] = (uint8_t)((t[0] >> 8) | (t[1] << 5));
        r[13*i+ 2] = (uint8_t)((t[1] >> 3));
        r[13*i+ 3] = (uint8_t)((t[1] >> 11) | (t[2] << 2));
        r[13*i+ 4] = (uint8_t)((t[2] >> 6)  | (t[3] << 7));
        r[13*i+ 5] = (uint8_t)((t[3] >> 1));
        r[13*i+ 6] = (uint8_t)((t[3] >> 9)  | (t[4] << 4));
        r[13*i+ 7] = (uint8_t)((t[4] >> 4));
        r[13*i+ 8] = (uint8_t)((t[4] >> 12) | (t[5] << 1));
        r[13*i+ 9] = (uint8_t)((t[5] >> 7)  | (t[6] << 6));
        r[13*i+10] = (uint8_t)((t[6] >> 2));
        r[13*i+11] = (uint8_t)((t[6] >> 10) | (t[7] << 3));
        r[13*i+12] = (uint8_t)((t[7] >> 5));
    }
}

static __device__ __noinline__ void poly_frombytes(kem_poly *r, const uint8_t *a)
{
    for (unsigned int i = 0; i < PARAM_N / 8; i++) {
        r->coeffs[8*i+0] = (int16_t)((a[13*i+ 0]       | ((uint16_t)a[13*i+ 1] << 8)) & 0x1FFF);
        r->coeffs[8*i+1] = (int16_t)(((a[13*i+ 1] >> 5) | ((uint16_t)a[13*i+ 2] << 3) | ((uint16_t)a[13*i+ 3] << 11)) & 0x1FFF);
        r->coeffs[8*i+2] = (int16_t)(((a[13*i+ 3] >> 2) | ((uint16_t)a[13*i+ 4] << 6)) & 0x1FFF);
        r->coeffs[8*i+3] = (int16_t)(((a[13*i+ 4] >> 7) | ((uint16_t)a[13*i+ 5] << 1) | ((uint16_t)a[13*i+ 6] << 9)) & 0x1FFF);
        r->coeffs[8*i+4] = (int16_t)(((a[13*i+ 6] >> 4) | ((uint16_t)a[13*i+ 7] << 4) | ((uint16_t)a[13*i+ 8] << 12)) & 0x1FFF);
        r->coeffs[8*i+5] = (int16_t)(((a[13*i+ 8] >> 1) | ((uint16_t)a[13*i+ 9] << 7)) & 0x1FFF);
        r->coeffs[8*i+6] = (int16_t)(((a[13*i+ 9] >> 6) | ((uint16_t)a[13*i+10] << 2) | ((uint16_t)a[13*i+11] << 10)) & 0x1FFF);
        r->coeffs[8*i+7] = (int16_t)(((a[13*i+11] >> 3) | ((uint16_t)a[13*i+12] << 5)) & 0x1FFF);
    }
}

#endif  /* ALGORITHM for tobytes/frombytes */

/* ================================================================
 *  密文标量多项式压缩/解压缩 (BITS_C2 bits per coeff)
 * ================================================================ */

/* 压缩: coeff in [0,Q) → BITS_C2-bit integer */
static __device__ __noinline__ void poly_compress_c2(uint8_t *r, const kem_poly *a)
{
    /* 先归一化到 [0, Q) */

#if PARAM_BITS_C2 == 3
    /* 3-bit: 8 coeffs → 3 bytes */
    for (int i = 0; i < PARAM_N / 8; i++) {
        uint8_t c0 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+0]) << 3) + PARAM_Q/2) / PARAM_Q) & 0x07;
        uint8_t c1 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+1]) << 3) + PARAM_Q/2) / PARAM_Q) & 0x07;
        uint8_t c2 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+2]) << 3) + PARAM_Q/2) / PARAM_Q) & 0x07;
        uint8_t c3 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+3]) << 3) + PARAM_Q/2) / PARAM_Q) & 0x07;
        uint8_t c4 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+4]) << 3) + PARAM_Q/2) / PARAM_Q) & 0x07;
        uint8_t c5 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+5]) << 3) + PARAM_Q/2) / PARAM_Q) & 0x07;
        uint8_t c6 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+6]) << 3) + PARAM_Q/2) / PARAM_Q) & 0x07;
        uint8_t c7 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+7]) << 3) + PARAM_Q/2) / PARAM_Q) & 0x07;
        r[3*i+0] = (uint8_t)(c0 | (c1 << 3) | (c2 << 6));
        r[3*i+1] = (uint8_t)((c2 >> 2) | (c3 << 1) | (c4 << 4) | (c5 << 7));
        r[3*i+2] = (uint8_t)((c5 >> 1) | (c6 << 2) | (c7 << 5));
    }
#elif PARAM_BITS_C2 == 4
    /* 4-bit: 2 coeffs per byte */
    for (int i = 0; i < PARAM_N / 2; i++) {
        int16_t u = (int16_t)((((int32_t)caddq(a->coeffs[2*i])   << 4) + PARAM_Q/2) / PARAM_Q) & 0x0F;
        int16_t v = (int16_t)((((int32_t)caddq(a->coeffs[2*i+1]) << 4) + PARAM_Q/2) / PARAM_Q) & 0x0F;
        r[i] = (uint8_t)(u | (v << 4));
    }
#elif PARAM_BITS_C2 == 5
    /* 5-bit: 8 coeffs → 5 bytes */
    for (int i = 0; i < PARAM_N / 8; i++) {
        uint8_t c0 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+0]) << 5) + PARAM_Q/2) / PARAM_Q) & 0x1F;
        uint8_t c1 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+1]) << 5) + PARAM_Q/2) / PARAM_Q) & 0x1F;
        uint8_t c2 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+2]) << 5) + PARAM_Q/2) / PARAM_Q) & 0x1F;
        uint8_t c3 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+3]) << 5) + PARAM_Q/2) / PARAM_Q) & 0x1F;
        uint8_t c4 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+4]) << 5) + PARAM_Q/2) / PARAM_Q) & 0x1F;
        uint8_t c5 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+5]) << 5) + PARAM_Q/2) / PARAM_Q) & 0x1F;
        uint8_t c6 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+6]) << 5) + PARAM_Q/2) / PARAM_Q) & 0x1F;
        uint8_t c7 = (uint8_t)((((int32_t)caddq(a->coeffs[8*i+7]) << 5) + PARAM_Q/2) / PARAM_Q) & 0x1F;
        r[5*i+0] = (uint8_t)(c0 | (c1 << 5));
        r[5*i+1] = (uint8_t)((c1 >> 3) | (c2 << 2) | (c3 << 7));
        r[5*i+2] = (uint8_t)((c3 >> 1) | (c4 << 4));
        r[5*i+3] = (uint8_t)((c4 >> 4) | (c5 << 1) | (c6 << 6));
        r[5*i+4] = (uint8_t)((c6 >> 2) | (c7 << 3));
    }
#endif
}

/* 解压缩: BITS_C2-bit integer → coeff in [0, Q) */
static __device__ __noinline__ void poly_decompress_c2(kem_poly *r, const uint8_t *a)
{
#if PARAM_BITS_C2 == 3
    for (int i = 0; i < PARAM_N / 8; i++) {
        uint8_t c[8];
        c[0] =  a[3*i+0]       & 0x07;
        c[1] = (a[3*i+0] >> 3) & 0x07;
        c[2] = (a[3*i+0] >> 6) | ((a[3*i+1] & 0x01) << 2);
        c[3] = (a[3*i+1] >> 1) & 0x07;
        c[4] = (a[3*i+1] >> 4) & 0x07;
        c[5] = (a[3*i+1] >> 7) | ((a[3*i+2] & 0x03) << 1);
        c[6] = (a[3*i+2] >> 2) & 0x07;
        c[7] = (a[3*i+2] >> 5);
        for (int j = 0; j < 8; j++)
            r->coeffs[8*i+j] = (int16_t)(((int32_t)c[j] * PARAM_Q + 4) >> 3);
    }
#elif PARAM_BITS_C2 == 4
    for (int i = 0; i < PARAM_N / 2; i++) {
        r->coeffs[2*i]   = (int16_t)(((int32_t)( a[i]       & 0x0F) * PARAM_Q + 8) >> 4);
        r->coeffs[2*i+1] = (int16_t)(((int32_t)((a[i] >> 4) & 0x0F) * PARAM_Q + 8) >> 4);
    }
#elif PARAM_BITS_C2 == 5
    for (int i = 0; i < PARAM_N / 8; i++) {
        uint8_t c[8];
        c[0] =  a[5*i+0]       & 0x1F;
        c[1] = (a[5*i+0] >> 5) | ((a[5*i+1] & 0x03) << 3);
        c[2] = (a[5*i+1] >> 2) & 0x1F;
        c[3] = (a[5*i+1] >> 7) | ((a[5*i+2] & 0x0F) << 1);
        c[4] = (a[5*i+2] >> 4) | ((a[5*i+3] & 0x01) << 4);
        c[5] = (a[5*i+3] >> 1) & 0x1F;
        c[6] = (a[5*i+3] >> 6) | ((a[5*i+4] & 0x07) << 2);
        c[7] =  a[5*i+4] >> 3;
        for (int j = 0; j < 8; j++)
            r->coeffs[8*i+j] = (int16_t)(((int32_t)c[j] * PARAM_Q + 16) >> 5);
    }
#endif
}

#endif /* POLY_CUH */
