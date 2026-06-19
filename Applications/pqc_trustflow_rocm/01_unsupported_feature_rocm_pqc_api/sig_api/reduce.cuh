/*
 * reduce.cuh — 统一约化函数
 *
 * 对两种算法均使用 int32_t 系数。
 * montgomery_reduce 使用无符号乘法取低32位的技巧规避有符号溢出 UB,
 * 数学结果与标准 Montgomery 约化完全等价。
 *
 * ML-DSA: 系数中心化 (-Q/2, Q/2], 使用 reduce32 + caddq
 * Aigis:  系数无符号 [0, Q), 使用 freeze2q/freeze4q
 */

#ifndef REDUCE_CUH
#define REDUCE_CUH

#include <stdint.h>
#include "params.h"

/*
 * montgomery_reduce(a)
 *   输入: a ∈ (-Q*2^32, Q*2^32)
 *   输出: a * R^{-1} mod Q, 结果 ∈ (-Q, Q)  (R = 2^32)
 *
 * 对 ML-DSA 和 Aigis 均正确。
 */
static __device__ __forceinline__ int32_t montgomery_reduce(int64_t a) {
    uint32_t t = (uint32_t)(int32_t)a * MONT_QINV;   /* uint32 wraparound: defined */
    return (int32_t)((a - (int64_t)t * PARAM_Q) >> 32);
}

/* ---- ML-DSA centered reduction ---- */

/*
 * reduce32(a): 中心化约化至 (-Q/2, Q/2]
 */
static __device__ __forceinline__ int32_t reduce32(int32_t a) {
    int32_t t = (a + (1 << (PARAM_QBITS - 1))) >> PARAM_QBITS;
    return a - t * PARAM_Q;
}

/*
 * caddq(a): 如果 a < 0, 加上 Q, 使其进入 [0, Q)
 */
static __device__ __forceinline__ int32_t caddq(int32_t a) {
    a += (a >> 31) & PARAM_Q;
    return a;
}

/*
 * freeze(a): 完全约化至 [0, Q)  (reduce32 + caddq)
 */
static __device__ __forceinline__ int32_t freeze(int32_t a) {
    return caddq(reduce32(a));
}

/* ---- Aigis unsigned reduction [0, Q) ---- */
/* GPU 使用有符号 Montgomery 运算, 中间值可能为负数。
 * 因此 freeze2q/freeze4q 必须先处理负值输入。
 * freeze2q: 输入 a ∈ (-2Q, 2Q), 输出 [0, Q)
 * freeze4q: 输入 a ∈ (-4Q, 4Q), 输出 [0, Q)
 */

static __device__ __forceinline__ int32_t freeze2q(int32_t a) {
    a += (a >> 31) & (2 * PARAM_Q);   /* 负值加 2Q → [0, 4Q) */
    a -= PARAM_Q;
    a += (a >> 31) & PARAM_Q;
    return a;
}

static __device__ __forceinline__ int32_t freeze4q(int32_t a) {
    a += (a >> 31) & (4 * PARAM_Q);   /* 负值加 4Q → [0, 8Q) */
    a -= 2 * PARAM_Q;
    a += (a >> 31) & (2 * PARAM_Q);
    a -= PARAM_Q;
    a += (a >> 31) & PARAM_Q;
    return a;
}

#if ALGORITHM == ALGO_AIGIS
/*
 * barrat_reduce(a): GPU 有符号版 — 使用 reduce32 保证正确性
 * 输入: 任意 int32_t, 输出: [0, 2Q) 大致
 */
static __device__ __forceinline__ int32_t barrat_reduce(int32_t a) {
    /* reduce32 在 GPU 有符号算术下安全, 输出 (-Q/2, Q/2] */
    return caddq(reduce32(a));
}
#endif /* ALGO_AIGIS */

/* ================================================================
 *  统一系数运算包装 — batch kernel 使用
 *  通过 coeff_t / coeff2_t 实现类型无关的批量运算
 * ================================================================ */

/* Montgomery multiply: c = a * b * R^{-1} mod Q */
static __device__ __forceinline__ coeff_t coeff_fqmul(coeff_t a, coeff_t b) {
    return montgomery_reduce((coeff2_t)a * b);
}

/* 模减法: 保持在 lazy-reduced 范围 */
static __device__ __forceinline__ coeff_t coeff_sub(coeff_t a, coeff_t b) {
#if ALGORITHM == ALGO_AIGIS
    /* Aigis 使用 int32_t 但系数 [0,Q), 减法后可能为负 → 加 2Q 保正 */
    return a + 2 * PARAM_Q - b;
#else
    return a - b;   /* ML-DSA: signed 直接减 */
#endif
}

/* 轻量约化至 ~(-Q, Q) 或 ~[0, 2Q) */
static __device__ __forceinline__ coeff_t coeff_reduce(coeff_t a) {
#if ALGORITHM == ALGO_AIGIS
    return barrat_reduce(a);
#else
    return reduce32(a);
#endif
}

/* 归一化至 [0, Q) */
static __device__ __forceinline__ coeff_t coeff_normalize(coeff_t a) {
#if ALGORITHM == ALGO_AIGIS
    return freeze2q(a);
#else
    return caddq(a);
#endif
}

/* 宽范围归一化至 [0, Q) */
static __device__ __forceinline__ coeff_t coeff_freeze_wide(coeff_t a) {
#if ALGORITHM == ALGO_AIGIS
    return freeze4q(a);
#else
    return caddq(reduce32(a));
#endif
}

#endif /* REDUCE_CUH */
