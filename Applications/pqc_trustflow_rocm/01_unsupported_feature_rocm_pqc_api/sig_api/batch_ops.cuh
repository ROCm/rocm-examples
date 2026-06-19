#include "hip/hip_runtime.h"
/*
 * batch_ops.cuh — 统一的批量多项式算术 kernel
 *
 * 逐系数操作: 每个线程处理一个系数, 256 threads/block.
 * 通过 coeff_t / coeff_* 包装函数实现算法无关.
 */

#ifndef BATCH_OPS_CUH
#define BATCH_OPS_CUH

#include <hip/hip_runtime.h>
#include <stdint.h>
#include "params.h"
#include "reduce.cuh"
#include "rounding.cuh"

#define BATCH_TPB 256

/* ================================================================
 * 共用 kernel — 两种算法一份代码
 * ================================================================ */

__global__ void batch_poly_add_kernel(coeff_t *c, const coeff_t *a,
                                      const coeff_t *b, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) c[idx] = a[idx] + b[idx];
}

__global__ void batch_poly_sub_kernel(coeff_t *c, const coeff_t *a,
                                      const coeff_t *b, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) c[idx] = coeff_sub(a[idx], b[idx]);
}

__global__ void batch_poly_pointwise_kernel(coeff_t *c, const coeff_t *a,
                                            const coeff_t *b, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) c[idx] = coeff_fqmul(a[idx], b[idx]);
}

__global__ void batch_poly_reduce_kernel(coeff_t *a, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) a[idx] = coeff_reduce(a[idx]);
}

__global__ void batch_poly_normalize_kernel(coeff_t *a, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) a[idx] = coeff_normalize(a[idx]);
}

__global__ void batch_poly_freeze_wide_kernel(coeff_t *a, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) a[idx] = coeff_freeze_wide(a[idx]);
}

__global__ void batch_poly_shiftl_kernel(coeff_t *a, int total, unsigned int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) a[idx] <<= k;
}

/* ================================================================
 * power2round kernel — 算法差异已在 rounding.cuh 中封装
 * ================================================================ */
__global__ void batch_power2round_kernel(coeff_t *d_a1, coeff_t *d_a0,
                                         const coeff_t *d_a, int total_coeffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) return;

#if ALGORITHM == ALGO_MLDSA
    coeff_t val = d_a[idx];
    /* ML-DSA caddq before power2round */
    val += (val >> 31) & PARAM_Q;
    int32_t a0_val;
    d_a1[idx] = power2round(&a0_val, val);
    d_a0[idx] = a0_val;
#elif ALGORITHM == ALGO_AIGIS
    int32_t a0_val;
    d_a1[idx] = power2round(&a0_val, d_a[idx]);
    d_a0[idx] = a0_val;
#endif
}

/* ================================================================
 * use_hint kernel — 用于 verify pipeline
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

__global__ void batch_use_hint_kernel(coeff_t * __restrict__ d_out,
                                      const coeff_t * __restrict__ d_a,
                                      const coeff_t * __restrict__ d_hint,
                                      int total_coeffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) return;

    int32_t a = d_a[idx];
    int32_t hint = d_hint[idx];

    int32_t a1;
    a1 = (a + 127) >> 7;
#if PARAM_GAMMA2 == ((PARAM_Q-1)/32)
    a1 = (a1*1025 + (1 << 21)) >> 22;
    a1 &= 15;
#elif PARAM_GAMMA2 == ((PARAM_Q-1)/88)
    a1 = (a1*11275 + (1 << 23)) >> 24;
    a1 ^= ((43 - a1) >> 31) & a1;
#endif
    int32_t a0 = a - a1 * 2 * PARAM_GAMMA2;
    a0 -= (((PARAM_Q-1)/2 - a0) >> 31) & PARAM_Q;

    if (hint == 0) { d_out[idx] = a1; return; }

#if PARAM_GAMMA2 == ((PARAM_Q-1)/32)
    if (a0 > 0) d_out[idx] = (a1 + 1) & 15;
    else        d_out[idx] = (a1 - 1) & 15;
#elif PARAM_GAMMA2 == ((PARAM_Q-1)/88)
    if (a0 > 0) d_out[idx] = (a1 == 43) ?  0 : a1 + 1;
    else        d_out[idx] = (a1 ==  0) ? 43 : a1 - 1;
#endif
}

#elif ALGORITHM == ALGO_AIGIS

__global__ void batch_use_hint_kernel(coeff_t * __restrict__ d_out,
                                      const coeff_t * __restrict__ d_a,
                                      const coeff_t * __restrict__ d_hint,
                                      int total_coeffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) return;

    int32_t hint = d_hint[idx];
    int32_t a = d_a[idx];

    /* Aigis use_hint: call rounding.cuh use_hint directly */
    d_out[idx] = use_hint(a, hint);
}

#endif /* ALGORITHM */

/* ================================================================
 * Host launch wrappers
 * ================================================================ */

static inline void launch_batch_add(coeff_t *c, const coeff_t *a,
                                    const coeff_t *b, int total_coeffs,
                                    hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_poly_add_kernel<<<nblk, BATCH_TPB, 0, stream>>>(c, a, b, total_coeffs);
}

static inline void launch_batch_sub(coeff_t *c, const coeff_t *a,
                                    const coeff_t *b, int total_coeffs,
                                    hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_poly_sub_kernel<<<nblk, BATCH_TPB, 0, stream>>>(c, a, b, total_coeffs);
}

static inline void launch_batch_reduce(coeff_t *a, int total_coeffs,
                                       hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_poly_reduce_kernel<<<nblk, BATCH_TPB, 0, stream>>>(a, total_coeffs);
}

static inline void launch_batch_normalize(coeff_t *a, int total_coeffs,
                                          hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_poly_normalize_kernel<<<nblk, BATCH_TPB, 0, stream>>>(a, total_coeffs);
}

static inline void launch_batch_freeze_wide(coeff_t *a, int total_coeffs,
                                            hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_poly_freeze_wide_kernel<<<nblk, BATCH_TPB, 0, stream>>>(a, total_coeffs);
}

static inline void launch_batch_shiftl(coeff_t *a, int total_coeffs,
                                       unsigned int k, hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_poly_shiftl_kernel<<<nblk, BATCH_TPB, 0, stream>>>(a, total_coeffs, k);
}

static inline void launch_batch_power2round(coeff_t *v1, coeff_t *v0,
                                            const coeff_t *v, int total_coeffs,
                                            hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_power2round_kernel<<<nblk, BATCH_TPB, 0, stream>>>(v1, v0, v, total_coeffs);
}

static inline void launch_batch_use_hint(coeff_t *w, const coeff_t *u,
                                         const coeff_t *h, int total_coeffs,
                                         hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_use_hint_kernel<<<nblk, BATCH_TPB, 0, stream>>>(w, u, h, total_coeffs);
}

/* 别名: freeze2q / caddq → normalize, freeze4q → freeze_wide */
static inline void launch_batch_freeze2q(coeff_t *a, int poly_count,
                                         hipStream_t stream = 0) {
    launch_batch_normalize(a, poly_count * PARAM_N, stream);
}

static inline void launch_batch_caddq(coeff_t *a, int total_coeffs,
                                      hipStream_t stream = 0) {
    launch_batch_normalize(a, total_coeffs, stream);
}

#endif /* BATCH_OPS_CUH */
