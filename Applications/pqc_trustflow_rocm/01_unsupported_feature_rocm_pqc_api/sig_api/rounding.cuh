/*
 * rounding.cuh
 *
 * ML-DSA: 中心化系数 (-Q/2, Q/2]
 * Aigis:  无符号系数 [0, Q), a0 存为 Q+t 偏置形式
 */

#ifndef ROUNDING_CUH
#define ROUNDING_CUH

#include <stdint.h>
#include "params.h"
#include "reduce.cuh"

/* ================================================================
 *  power2round
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __forceinline__ int32_t power2round(int32_t *a0, int32_t a) {
    int32_t a1 = (a + (1 << (PARAM_D - 1)) - 1) >> PARAM_D;
    *a0 = a - (a1 << PARAM_D);
    return a1;
}

#elif ALGORITHM == ALGO_AIGIS

/* Aigis: unsigned input a ∈ [0,Q), output a0 = Q + t (biased), a1 = (a-t)>>D */
static __device__ __forceinline__ int32_t power2round(int32_t *a0, int32_t a) {
    int32_t t;
    t = a & ((1 << PARAM_D) - 1);
    t -= (1 << (PARAM_D - 1)) + 1;
    t += (t >> 31) & (1 << PARAM_D);
    t -= (1 << (PARAM_D - 1)) - 1;
    *a0 = PARAM_Q + t;
    a = (a - t) >> PARAM_D;
    return a;
}

#endif /* power2round */

/* ================================================================
 *  decompose
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __forceinline__ int32_t decompose(int32_t *a0, int32_t a) {
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

/* Aigis: unsigned a ∈ [0,Q), ALPHA=2*GAMMA2, (Q-1)=6*ALPHA
 * Output: a1 ∈ [0, N_W1), a0 = Q + t (biased, centered around Q) */
static __device__ __forceinline__ int32_t decompose(int32_t *a0, int32_t a) {
    int32_t t, u;
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
    if (a1 == N_W1) { *a0 = PARAM_Q + t - 1; a1 = 0; }
    else            { *a0 = PARAM_Q + t; }
    return a1;
}

#endif /* decompose */

/* ================================================================
 *  make_hint / use_hint
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __forceinline__ int32_t make_hint(int32_t a0, int32_t a1) {
    if (a0 > PARAM_GAMMA2 || a0 < -PARAM_GAMMA2 ||
        (a0 == -PARAM_GAMMA2 && a1 != 0))
        return 1;
    return 0;
}

static __device__ __forceinline__ int32_t use_hint(int32_t a, int32_t hint) {
    int32_t a0, a1;
    a1 = decompose(&a0, a);
    if (hint == 0) return a1;
    if (a0 > 0) return (a1 + 1 >= N_W1) ? 0 : a1 + 1;
    else        return (a1 - 1 < 0)     ? N_W1 - 1 : a1 - 1;
}

#elif ALGORITHM == ALGO_AIGIS

/* Aigis make_hint: comparison-based — hint=1 iff decompose(a) ≠ decompose(freeze4q(a+b)) */
static __device__ __forceinline__ int32_t make_hint(int32_t a, int32_t b) {
    int32_t t;
    return decompose(&t, a) != decompose(&t, freeze4q(a + b));
}

/* Aigis use_hint: unsigned a ∈ [0,Q), check a0 > Q (means centered a0 was negative) */
static __device__ __forceinline__ int32_t use_hint(int32_t a, int32_t hint) {
    int32_t a0, a1;
    a1 = decompose(&a0, a);
    if (hint == 0) return a1;
    if (a0 > PARAM_Q)
        return (a1 == (PARAM_Q - 1) / (2 * PARAM_GAMMA2) - 1) ? 0 : a1 + 1;
    else
        return (a1 == 0) ? (PARAM_Q - 1) / (2 * PARAM_GAMMA2) - 1 : a1 - 1;
}

#endif /* make_hint / use_hint */

#endif /* ROUNDING_CUH */
