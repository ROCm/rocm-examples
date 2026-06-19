/*
 * batch_ops.cuh — 批量多项式运算 kernel
 *
 * 256 threads/block, 每 thread 处理 1 个系数
 * SoA 内存布局: data[poly_idx * batch_count * N + inst * N + coeff]
 *
 * 提供的 kernel:
 *   batch_poly_add_kernel      — 向量加法
 *   batch_poly_sub_kernel      — 向量减法
 *   batch_poly_reduce_kernel   — Barrett 约减
 *   batch_poly_caddq_kernel    — 加 Q 规范化到 [0,Q)
 *   batch_poly_caddq2_kernel   — 双重 caddq 规范化
 *   batch_polyvec_matvec_acc_kernel — 矩阵向量乘 (含 basemul 或逐点乘)
 */

#ifndef BATCH_OPS_CUH
#define BATCH_OPS_CUH

#include "rocm_compat.h"
#include <stdint.h>
#include "params.h"
#include "reduce.cuh"
#include "ntt.cuh"

#define BATCH_TPB  256   /* threads per block for batch poly ops */

/* ================================================================
 *  基础逐系数运算
 *  SoA 格式: arrays[inst_idx * N + coeff_idx] (单多项式批次)
 * ================================================================ */

__global__ void batch_poly_add_kernel(
    int16_t * __restrict__ c,        /* output */
    const int16_t * __restrict__ a,  /* input 1 */
    const int16_t * __restrict__ b,  /* input 2 */
    int batch_count)
{
    int idx = blockIdx.x * BATCH_TPB + threadIdx.x;
    if (idx < batch_count * PARAM_N)
        c[idx] = a[idx] + b[idx];
}

__global__ void batch_poly_sub_kernel(
    int16_t * __restrict__ c,
    const int16_t * __restrict__ a,
    const int16_t * __restrict__ b,
    int batch_count)
{
    int idx = blockIdx.x * BATCH_TPB + threadIdx.x;
    if (idx < batch_count * PARAM_N)
        c[idx] = a[idx] - b[idx];
}

__global__ void batch_poly_reduce_kernel(
    int16_t * __restrict__ r,
    int batch_count)
{
    int idx = blockIdx.x * BATCH_TPB + threadIdx.x;
    if (idx < batch_count * PARAM_N)
        r[idx] = barrett_reduce(r[idx]);
}

__global__ void batch_poly_caddq_kernel(
    int16_t * __restrict__ r,
    int batch_count)
{
    int idx = blockIdx.x * BATCH_TPB + threadIdx.x;
    if (idx < batch_count * PARAM_N)
        r[idx] = caddq(r[idx]);
}

__global__ void batch_poly_caddq2_kernel(
    int16_t * __restrict__ r,
    int batch_count)
{
    int idx = blockIdx.x * BATCH_TPB + threadIdx.x;
    if (idx < batch_count * PARAM_N)
        r[idx] = caddq2(r[idx]);
}

/* ================================================================
 *  批量矩阵向量乘 (2D grid: gridDim.x=batch_count, gridDim.y=K)
 *
 *  SoA 格式:
 *    mat[row * K * batch_count * N + col * batch_count * N + inst * N + c]
 *    vec[col * batch_count * N + inst * N + c]
 *    out[row * batch_count * N + inst * N + c]
 *
 *  每个 block 处理一个 (batch_inst, row) 的输出多项式的一个系数
 *  threadIdx.x: 系数索引 (0..N-1), 256 threads
 *
 *  内积方式:
 *    Kyber:     basemul (4 coeffs at a time with ±zeta)
 *    Aigis-enc: pointwise fqmul (直接逐点乘)
 * ================================================================ */

__global__ void batch_polyvec_matvec_kernel(
    int16_t * __restrict__ d_out,          /* K * B * N */
    const int16_t * __restrict__ d_mat,    /* K * K * B * N, SoA */
    const int16_t * __restrict__ d_vec,    /* K * B * N, SoA */
    int batch_count)
{
    int inst = blockIdx.x;   /* 批次实例索引 */
    int row  = blockIdx.y;   /* 输出行 (0..K-1) */
    int c    = threadIdx.x;  /* 系数索引 (0..N-1, 256 threads) */

    if (inst >= batch_count) return;

#if ALGORITHM == ALGO_KYBER

    /* Kyber basemul 域内积
     * NTT 最后一级产生 64 组四元素 [4i,4i+1,4i+2,4i+3]:
     *   pair0: indices [4i, 4i+1] in Z_q[x]/(x^2 - zeta[64+i])
     *   pair1: indices [4i+2,4i+3] in Z_q[x]/(x^2 + zeta[64+i])
     * basemul: r[0] = fqmul(fqmul(a1,b1),zeta) + fqmul(a0,b0)
     *          r[1] = fqmul(a0,b1) + fqmul(a1,b0) */

    int quad  = c >> 2;        /* group i = c/4, 0..63 */
    int local = c & 3;         /* 0,1,2,3 within 4-group */
    int c_even = c & ~1;       /* floor to even: 4i or 4i+2 */
    int c_odd  = c | 1;        /* ceil to odd: 4i+1 or 4i+3 */
    /* zeta: +zeta[64+quad] for local=0,1; -zeta[64+quad] for local=2,3 */
    int16_t zeta_raw = ntt_zetas[64 + quad];
    int16_t zeta = (local < 2) ? zeta_raw : (int16_t)(-zeta_raw);

    int16_t acc = 0;
    for (int col = 0; col < PARAM_K; col++) {
        size_t base_m = ((size_t)(row * PARAM_K + col) * batch_count + inst) * PARAM_N;
        size_t base_v = ((size_t)col * batch_count + inst) * PARAM_N;
        int16_t a0 = d_mat[base_m + c_even];
        int16_t a1 = d_mat[base_m + c_odd];
        int16_t b0 = d_vec[base_v + c_even];
        int16_t b1 = d_vec[base_v + c_odd];

        if (local & 1) {
            /* r[c_odd] = a0*b1 + a1*b0 */
            acc = (int16_t)(acc + fqmul(a0, b1) + fqmul(a1, b0));
        } else {
            /* r[c_even] = fqmul(fqmul(a1,b1), zeta) + fqmul(a0,b0) */
            acc = (int16_t)(acc + fqmul(fqmul(a1, b1), zeta) + fqmul(a0, b0));
        }
    }
    d_out[((size_t)(row * batch_count) + inst) * PARAM_N + c] = barrett_reduce(acc);

#elif ALGORITHM == ALGO_AIGIS_ENC

    /* Aigis: 逐点乘, 累加 K 项后 Montgomery 约减一次
     * 最大值: K * Q^2 = 4 * 7681^2 ≈ 236M < 2^31 (int32_t 安全) */
    int32_t acc = 0;
    for (int col = 0; col < PARAM_K; col++) {
        int16_t av = d_mat[((size_t)(row * PARAM_K + col) * batch_count + inst) * PARAM_N + c];
        int16_t bv = d_vec[((size_t)col * batch_count + inst) * PARAM_N + c];
        acc += (int32_t)av * bv;
    }
    d_out[((size_t)(row * batch_count) + inst) * PARAM_N + c] = montgomery_reduce(acc);

#endif
}

/* ================================================================
 *  批量 frommsg / tomsg kernel (SoA: [inst * N + coeff])
 * ================================================================ */

__global__ void batch_poly_frommsg_kernel(
    int16_t * __restrict__ d_poly,    /* B * N */
    const uint8_t * __restrict__ d_msgs, /* B * N/8 */
    int batch_count)
{
    int inst = blockIdx.x;
    int c    = threadIdx.x;  /* 0..N-1 */
    if (inst >= batch_count) return;

    int byte_idx = c >> 3;
    int bit_idx  = c & 7;
    uint8_t bit = (d_msgs[inst * (PARAM_N / 8) + byte_idx] >> bit_idx) & 1;
    int16_t mask = -(int16_t)bit;  /* 0x0000 or 0xFFFF */
    d_poly[inst * PARAM_N + c] = mask & (int16_t)((PARAM_Q + 1) / 2);
}

__global__ void batch_poly_tomsg_kernel(
    uint8_t * __restrict__ d_msgs,      /* B * N/8 */
    const int16_t * __restrict__ d_poly, /* B * N */
    int batch_count)
{
    int inst = blockIdx.x;
    int c    = threadIdx.x;  /* 0..N-1 */
    if (inst >= batch_count) return;

    int16_t t = d_poly[inst * PARAM_N + c];
    t = caddq(t);
    uint8_t bit = (uint8_t)(((((int32_t)t << 1) + PARAM_Q / 2 + 1) / PARAM_Q) & 1);

    /* 原子或写入 bit */
    int byte_idx = c >> 3;
    int bit_idx  = c & 7;
    atomicOr((unsigned int *)(d_msgs + inst * (PARAM_N / 8) + (byte_idx & ~3)),
             (unsigned int)((unsigned int)bit << ((byte_idx & 3) * 8 + bit_idx)));
}

/* ================================================================
 *  Host 启动封装
 * ================================================================ */

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static inline void launch_batch_add(int16_t *d_c, const int16_t *d_a, const int16_t *d_b,
                                     int batch_count, cudaStream_t stream = 0) {
    batch_poly_add_kernel<<<ceil_div(batch_count * PARAM_N, BATCH_TPB), BATCH_TPB, 0, stream>>>(
        d_c, d_a, d_b, batch_count);
}

static inline void launch_batch_sub(int16_t *d_c, const int16_t *d_a, const int16_t *d_b,
                                     int batch_count, cudaStream_t stream = 0) {
    batch_poly_sub_kernel<<<ceil_div(batch_count * PARAM_N, BATCH_TPB), BATCH_TPB, 0, stream>>>(
        d_c, d_a, d_b, batch_count);
}

static inline void launch_batch_reduce(int16_t *d_r, int batch_count, cudaStream_t stream = 0) {
    batch_poly_reduce_kernel<<<ceil_div(batch_count * PARAM_N, BATCH_TPB), BATCH_TPB, 0, stream>>>(
        d_r, batch_count);
}

static inline void launch_batch_caddq(int16_t *d_r, int batch_count, cudaStream_t stream = 0) {
    batch_poly_caddq_kernel<<<ceil_div(batch_count * PARAM_N, BATCH_TPB), BATCH_TPB, 0, stream>>>(
        d_r, batch_count);
}

static inline void launch_batch_caddq2(int16_t *d_r, int batch_count, cudaStream_t stream = 0) {
    batch_poly_caddq2_kernel<<<ceil_div(batch_count * PARAM_N, BATCH_TPB), BATCH_TPB, 0, stream>>>(
        d_r, batch_count);
}

static inline void launch_batch_matvec(
    int16_t *d_out, const int16_t *d_mat, const int16_t *d_vec,
    int batch_count, cudaStream_t stream = 0)
{
    /* 2D grid: (batch_count, K), 256 threads */
    dim3 grid(batch_count, PARAM_K);
    batch_polyvec_matvec_kernel<<<grid, PARAM_N, 0, stream>>>(
        d_out, d_mat, d_vec, batch_count);
}

#endif /* BATCH_OPS_CUH */
