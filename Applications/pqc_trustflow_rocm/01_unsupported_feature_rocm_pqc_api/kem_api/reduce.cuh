/*
 * reduce.cuh — 统一模块化约减
 *
 * 两种算法均使用 16-bit Montgomery 约减 (R = 2^16)。
 * 唯一差异是 Q 和 QINV 的数值，由 params.h 提供。
 *
 * Montgomery 乘法: fqmul(a, b) = a*b*R^{-1} mod Q
 *   步骤:  t = a * QINV (mod 2^16, 有符号)
 *          return (a*b - t*Q) / R
 *
 * Barrett 约减 (仅 Aigis-enc 使用, Kyber 同样支持):
 *   输入范围 [-(2^15)*Q, (2^15)*Q], 输出 (-Q, Q)
 */

#ifndef REDUCE_CUH
#define REDUCE_CUH

#include <stdint.h>
#include "params.h"

/* ================================================================
 *  Montgomery 约减: 输入 int32_t a，输出 int16_t ≡ a*R^{-1} mod Q
 *  有效输入范围: |a| < Q * 2^15
 * ================================================================ */
static __device__ __forceinline__ int16_t montgomery_reduce(int32_t a)
{
    int16_t t = (int16_t)((int16_t)a * (int16_t)PARAM_QINV);
    return (int16_t)((a - (int32_t)t * PARAM_Q) >> 16);
}

/* Montgomery 乘法 */
static __device__ __forceinline__ int16_t fqmul(int16_t a, int16_t b)
{
    return montgomery_reduce((int32_t)a * b);
}

/* ================================================================
 *  Barrett 约减: 输入 int16_t a in (-Q*4, Q*4)，输出 (-Q, Q)
 *  使用预计算常数 v ≈ 2^26 / Q
 * ================================================================ */
static __device__ __forceinline__ int16_t barrett_reduce(int16_t a)
{
#if ALGORITHM == ALGO_KYBER
    /* Kyber Q=3329, v=(2^26 + Q/2)/Q = 20159 */
    const int16_t v = (int16_t)(((1 << 26) + PARAM_Q / 2) / PARAM_Q);
    int16_t t = (int16_t)(((int32_t)v * a + (1 << 25)) >> 26);
    return a - t * (int16_t)PARAM_Q;
#elif ALGORITHM == ALGO_AIGIS_ENC
    /* Aigis Q=7681, 使用 (a + 2^12) >> 13 * Q 近似 */
    int16_t u = (int16_t)((a + (1 << 12)) >> 13);
    u *= (int16_t)PARAM_Q;
    return a - u;
#endif
}

/* ================================================================
 *  caddq: 将 [-Q, Q) 规范化到 [0, Q)
 * ================================================================ */
static __device__ __forceinline__ int16_t caddq(int16_t a)
{
    return a + ((a >> 15) & (int16_t)PARAM_Q);
}

/* 双重 caddq: 将 [-2Q, Q) 映射到 [0, Q) */
static __device__ __forceinline__ int16_t caddq2(int16_t a)
{
    int16_t r = a + ((a >> 15) & (int16_t)PARAM_Q);
    return r + ((r >> 15) & (int16_t)PARAM_Q);
}

/* ================================================================
 *  tomont: 将普通系数转换到 Montgomery 域
 *  result = a * R^2 * R^{-1} = a * R  (mod Q)
 * ================================================================ */
static __device__ __forceinline__ int16_t tomont(int16_t a)
{
    return fqmul(a, (int16_t)MONT_R2);
}

#endif /* REDUCE_CUH */
