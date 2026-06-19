/*
 * polyvec.cuh — 统一多项式向量运算
 *
 * 包含:
 *   - polyvec_tobytes/frombytes (全精度序列化)
 *   - polyvec_pk_compress/decompress (公钥向量压缩)
 *   - polyvec_ct_compress/decompress (密文向量 u 压缩, BITS_C1 bits)
 *   - polyvec_ntt/invntt
 *   - polyvec_basemul_acc (内积, 由 ntt.cuh 提供)
 *   - polyvec_add, polyvec_caddq
 *
 * 公钥压缩位宽 PARAM_BITS_PK:
 *   Kyber: 12 bits (不压缩, 直接用 tobytes12)
 *   Aigis: 9, 10, or 11 bits
 *
 * 密文向量压缩位宽 PARAM_BITS_C1:
 *   Kyber: 10 bits (K=2,3) or 11 bits (K=4)
 *   Aigis: 9, 10, or 11 bits
 */

#ifndef POLYVEC_CUH
#define POLYVEC_CUH

#include <stdint.h>
#include "params.h"
#include "poly.cuh"
#include "ntt.cuh"

/* ================================================================
 *  向量级基础操作
 * ================================================================ */

static __device__ void polyvec_add(kem_polyvec *r, const kem_polyvec *a, const kem_polyvec *b)
{
    for (int i = 0; i < PARAM_K; i++) poly_add(&r->vec[i], &a->vec[i], &b->vec[i]);
}

static __device__ void polyvec_reduce(kem_polyvec *r)
{
    for (int i = 0; i < PARAM_K; i++) poly_reduce(&r->vec[i]);
}

static __device__ void polyvec_caddq(kem_polyvec *r)
{
    for (int i = 0; i < PARAM_K; i++) poly_caddq(&r->vec[i]);
}

static __device__ void polyvec_caddq2(kem_polyvec *r)
{
    for (int i = 0; i < PARAM_K; i++) poly_caddq2(&r->vec[i]);
}

/* ================================================================
 *  全精度序列化 (用于 sk 存储 NTT 域系数)
 * ================================================================ */

static __device__ __noinline__ void polyvec_tobytes(uint8_t *r, const kem_polyvec *a)
{
    for (int i = 0; i < PARAM_K; i++)
        poly_tobytes(r + i * PARAM_POLYBYTES, &a->vec[i]);
}

static __device__ __noinline__ void polyvec_frombytes(kem_polyvec *r, const uint8_t *a)
{
    for (int i = 0; i < PARAM_K; i++)
        poly_frombytes(&r->vec[i], a + i * PARAM_POLYBYTES);
}

/* ================================================================
 *  通用有损压缩辅助函数 (9/10/11 bits)
 *  使用 PARAM_Q — 同时适用于 Kyber 和 Aigis-enc CT 压缩
 * ================================================================ */

/* 9-bit 压缩: 8 coeffs → 9 bytes */
static __device__ __noinline__ void polyvec_compress9(uint8_t *r, const kem_poly *a)
{
    for (int i = 0; i < PARAM_N / 8; i++) {
        uint16_t c[8];
        for (int j = 0; j < 8; j++)
            c[j] = (uint16_t)((((int32_t)caddq(a->coeffs[8*i+j]) << 9) + PARAM_Q/2) / PARAM_Q) & 0x1FF;
        r[9*i+0] = (uint8_t)(c[0]);
        r[9*i+1] = (uint8_t)((c[0] >> 8) | (c[1] << 1));
        r[9*i+2] = (uint8_t)((c[1] >> 7) | (c[2] << 2));
        r[9*i+3] = (uint8_t)((c[2] >> 6) | (c[3] << 3));
        r[9*i+4] = (uint8_t)((c[3] >> 5) | (c[4] << 4));
        r[9*i+5] = (uint8_t)((c[4] >> 4) | (c[5] << 5));
        r[9*i+6] = (uint8_t)((c[5] >> 3) | (c[6] << 6));
        r[9*i+7] = (uint8_t)((c[6] >> 2) | (c[7] << 7));
        r[9*i+8] = (uint8_t)((c[7] >> 1));
    }
}

static __device__ __noinline__ void polyvec_decompress9(kem_poly *r, const uint8_t *a)
{
    for (int i = 0; i < PARAM_N / 8; i++) {
        uint16_t c[8];
        c[0] = ((uint16_t)a[9*i+0])       | ((uint16_t)(a[9*i+1] & 0x01) << 8);
        c[1] = ((uint16_t)a[9*i+1] >> 1)  | ((uint16_t)(a[9*i+2] & 0x03) << 7);
        c[2] = ((uint16_t)a[9*i+2] >> 2)  | ((uint16_t)(a[9*i+3] & 0x07) << 6);
        c[3] = ((uint16_t)a[9*i+3] >> 3)  | ((uint16_t)(a[9*i+4] & 0x0F) << 5);
        c[4] = ((uint16_t)a[9*i+4] >> 4)  | ((uint16_t)(a[9*i+5] & 0x1F) << 4);
        c[5] = ((uint16_t)a[9*i+5] >> 5)  | ((uint16_t)(a[9*i+6] & 0x3F) << 3);
        c[6] = ((uint16_t)a[9*i+6] >> 6)  | ((uint16_t)(a[9*i+7] & 0x7F) << 2);
        c[7] = ((uint16_t)a[9*i+7] >> 7)  | ((uint16_t)(a[9*i+8]) << 1);
        for (int j = 0; j < 8; j++)
            r->coeffs[8*i+j] = (int16_t)(((int32_t)c[j] * PARAM_Q + 256) >> 9);
    }
}

/* 10-bit 压缩: 4 coeffs → 5 bytes */
static __device__ __noinline__ void polyvec_compress10(uint8_t *r, const kem_poly *a)
{
    for (int i = 0; i < PARAM_N / 4; i++) {
        uint16_t c[4];
        for (int j = 0; j < 4; j++)
            c[j] = (uint16_t)((((int32_t)caddq(a->coeffs[4*i+j]) << 10) + PARAM_Q/2) / PARAM_Q) & 0x3FF;
        r[5*i+0] = (uint8_t)(c[0]);
        r[5*i+1] = (uint8_t)((c[0] >> 8) | (c[1] << 2));
        r[5*i+2] = (uint8_t)((c[1] >> 6) | (c[2] << 4));
        r[5*i+3] = (uint8_t)((c[2] >> 4) | (c[3] << 6));
        r[5*i+4] = (uint8_t)((c[3] >> 2));
    }
}

static __device__ __noinline__ void polyvec_decompress10(kem_poly *r, const uint8_t *a)
{
    for (int i = 0; i < PARAM_N / 4; i++) {
        uint16_t c[4];
        c[0] = ((uint16_t)a[5*i+0])      | ((uint16_t)(a[5*i+1] & 0x03) << 8);
        c[1] = ((uint16_t)a[5*i+1] >> 2) | ((uint16_t)(a[5*i+2] & 0x0F) << 6);
        c[2] = ((uint16_t)a[5*i+2] >> 4) | ((uint16_t)(a[5*i+3] & 0x3F) << 4);
        c[3] = ((uint16_t)a[5*i+3] >> 6) | ((uint16_t)(a[5*i+4]) << 2);
        for (int j = 0; j < 4; j++)
            r->coeffs[4*i+j] = (int16_t)(((int32_t)c[j] * PARAM_Q + 512) >> 10);
    }
}

/* 11-bit 压缩: 8 coeffs → 11 bytes */
static __device__ __noinline__ void polyvec_compress11(uint8_t *r, const kem_poly *a)
{
    for (int i = 0; i < PARAM_N / 8; i++) {
        uint16_t c[8];
        for (int j = 0; j < 8; j++)
            c[j] = (uint16_t)((((int32_t)caddq(a->coeffs[8*i+j]) << 11) + PARAM_Q/2) / PARAM_Q) & 0x7FF;
        r[11*i+ 0] = (uint8_t)(c[0]);
        r[11*i+ 1] = (uint8_t)((c[0] >> 8) | (c[1] << 3));
        r[11*i+ 2] = (uint8_t)((c[1] >> 5) | (c[2] << 6));
        r[11*i+ 3] = (uint8_t)((c[2] >> 2));
        r[11*i+ 4] = (uint8_t)((c[2] >> 10) | (c[3] << 1));
        r[11*i+ 5] = (uint8_t)((c[3] >> 7) | (c[4] << 4));
        r[11*i+ 6] = (uint8_t)((c[4] >> 4) | (c[5] << 7));
        r[11*i+ 7] = (uint8_t)((c[5] >> 1));
        r[11*i+ 8] = (uint8_t)((c[5] >> 9) | (c[6] << 2));
        r[11*i+ 9] = (uint8_t)((c[6] >> 6) | (c[7] << 5));
        r[11*i+10] = (uint8_t)((c[7] >> 3));
    }
}

static __device__ __noinline__ void polyvec_decompress11(kem_poly *r, const uint8_t *a)
{
    for (int i = 0; i < PARAM_N / 8; i++) {
        uint16_t c[8];
        c[0] = ((uint16_t)a[11*i+ 0])      | ((uint16_t)(a[11*i+ 1] & 0x07) << 8);
        c[1] = ((uint16_t)a[11*i+ 1] >> 3) | ((uint16_t)(a[11*i+ 2] & 0x3F) << 5);
        c[2] = ((uint16_t)a[11*i+ 2] >> 6) | ((uint16_t)a[11*i+ 3] << 2) | ((uint16_t)(a[11*i+ 4] & 0x01) << 10);
        c[3] = ((uint16_t)a[11*i+ 4] >> 1) | ((uint16_t)(a[11*i+ 5] & 0x0F) << 7);
        c[4] = ((uint16_t)a[11*i+ 5] >> 4) | ((uint16_t)(a[11*i+ 6] & 0x7F) << 4);
        c[5] = ((uint16_t)a[11*i+ 6] >> 7) | ((uint16_t)a[11*i+ 7] << 1) | ((uint16_t)(a[11*i+ 8] & 0x03) << 9);
        c[6] = ((uint16_t)a[11*i+ 8] >> 2) | ((uint16_t)(a[11*i+ 9] & 0x1F) << 6);
        c[7] = ((uint16_t)a[11*i+ 9] >> 5) | ((uint16_t)a[11*i+10] << 3);
        for (int j = 0; j < 8; j++)
            r->coeffs[8*i+j] = (int16_t)(((int32_t)c[j] * PARAM_Q + 1024) >> 11);
    }
}

/* ================================================================
 *  公钥向量压缩/解压缩 (PARAM_BITS_PK bits per coeff)
 *
 *  Kyber: BITS_PK=12 → tobytes (无压缩)
 *  Aigis: BITS_PK=9/10/11 → compress9/10/11
 * ================================================================ */

#if ALGORITHM == ALGO_KYBER
static __device__ __noinline__ void polyvec_pk_compress(uint8_t *r, const kem_polyvec *a)
{
    polyvec_tobytes(r, a);
}
static __device__ __noinline__ void polyvec_pk_decompress(kem_polyvec *r, const uint8_t *a)
{
    polyvec_frombytes(r, a);
}
#elif ALGORITHM == ALGO_AIGIS_ENC
/* 统一 PK 压缩/解压缩分发 */
static __device__ __noinline__ void polyvec_pk_compress(uint8_t *r, const kem_polyvec *a)
{
    for (int i = 0; i < PARAM_K; i++) {
        uint8_t *dst = r + i * PARAM_BITS_PK * PARAM_N / 8;
#if PARAM_BITS_PK == 9
        polyvec_compress9(dst, &a->vec[i]);
#elif PARAM_BITS_PK == 10
        polyvec_compress10(dst, &a->vec[i]);
#elif PARAM_BITS_PK == 11
        polyvec_compress11(dst, &a->vec[i]);
#endif
    }
}

static __device__ __noinline__ void polyvec_pk_decompress(kem_polyvec *r, const uint8_t *a)
{
    for (int i = 0; i < PARAM_K; i++) {
        const uint8_t *src = a + i * PARAM_BITS_PK * PARAM_N / 8;
#if PARAM_BITS_PK == 9
        polyvec_decompress9(&r->vec[i], src);
#elif PARAM_BITS_PK == 10
        polyvec_decompress10(&r->vec[i], src);
#elif PARAM_BITS_PK == 11
        polyvec_decompress11(&r->vec[i], src);
#endif
    }
}
#endif  /* ALGORITHM for PK compress */

/* ================================================================
 *  密文向量 u 压缩/解压缩 (PARAM_BITS_C1 bits per coeff)
 *
 *  两种算法都有 10-bit 和 11-bit 变体 (Aigis 还有 9-bit)
 *  Kyber K=2,3: 10-bit; K=4: 11-bit
 *  Aigis: 按 PARAM_BITS_C1 选择
 * ================================================================ */

static __device__ __noinline__ void polyvec_ct_compress(uint8_t *r, const kem_polyvec *a)
{
    for (int i = 0; i < PARAM_K; i++) {
        uint8_t *dst = r + i * PARAM_BITS_C1 * PARAM_N / 8;
#if PARAM_BITS_C1 == 9
        polyvec_compress9(dst, &a->vec[i]);
#elif PARAM_BITS_C1 == 10
        polyvec_compress10(dst, &a->vec[i]);
#elif PARAM_BITS_C1 == 11
        polyvec_compress11(dst, &a->vec[i]);
#endif
    }
}

static __device__ __noinline__ void polyvec_ct_decompress(kem_polyvec *r, const uint8_t *a)
{
    for (int i = 0; i < PARAM_K; i++) {
        const uint8_t *src = a + i * PARAM_BITS_C1 * PARAM_N / 8;
#if PARAM_BITS_C1 == 9
        polyvec_decompress9(&r->vec[i], src);
#elif PARAM_BITS_C1 == 10
        polyvec_decompress10(&r->vec[i], src);
#elif PARAM_BITS_C1 == 11
        polyvec_decompress11(&r->vec[i], src);
#endif
    }
}

#endif /* POLYVEC_CUH */
