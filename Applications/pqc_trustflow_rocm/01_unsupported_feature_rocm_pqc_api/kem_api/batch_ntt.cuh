/*
 * batch_ntt.cuh — GPU 批量 NTT/INVNTT
 *
 * 每个 block 处理一个多项式 (128 threads/block)
 * 使用共享内存避免全局内存延迟
 *
 * Kyber:     7 级蝶形 (len=128→2)，常数表 ntt_zetas[128]
 * Aigis-enc: 8 级蝶形 (len=128→1)，常数表 ntt_zetas[256] 和 ntt_zetas_inv[256]
 *
 * 批量 NTT 格式:
 *   polys[poly_idx * N + coeff_idx] — AoS 布局 (poly 连续, 每 poly N 个 int16_t)
 *   SoA: polys[poly_idx][batch_inst][coeff_idx] 由 batch_kem.cuh 处理
 *
 * 优化:
 *   共享内存 bank 填充: SP(i) = i + (i >> 5) 避免 bank 冲突 (32-bit bank)
 *   注: int16_t 使用 SP(i)  = i + (i >> 4) 可能更优, 但参考 mldsa 实现
 */

#ifndef BATCH_NTT_CUH
#define BATCH_NTT_CUH

#include "rocm_compat.h"
#include <stdint.h>
#include "params.h"
#include "reduce.cuh"
#include "ntt.cuh"

/* bank 填充宏: 将 256 int16_t 扩展为 264 元素以避免 bank 冲突
 * 实际: 每 32 个 int16_t 后插入 1 个 padding → S[i + (i>>5)] */
#define SP(i)  ((i) + ((i) >> 5))
#define SPAD   (PARAM_N + (PARAM_N >> 5))  /* 264 */

/* ================================================================
 *  批量 NTT kernel
 *  polys: int16_t 数组, 每 poly N 个连续系数
 *  batch_count: poly 个数
 * ================================================================ */
__global__ void batch_ntt_kernel(int16_t * __restrict__ polys, int batch_count)
{
    int poly_idx = blockIdx.x;
    if (poly_idx >= batch_count) return;

    int tid = (int)threadIdx.x;  /* 0..127 */

    __shared__ int16_t s[SPAD];

    /* 加载 poly 到共享内存 */
    int16_t *base = polys + poly_idx * PARAM_N;
    s[SP(tid)]       = base[tid];
    s[SP(tid + 128)] = base[tid + 128];
    __syncthreads();

#if ALGORITHM == ALGO_KYBER

    /* Kyber: 7 级, len=128→2, zeta 索引从 1 开始 */
    /* Level 7: len=128, 1 group */
    {
        int16_t zeta = ntt_zetas[1];
        int j = tid;  /* 0..127 */
        int16_t t = fqmul(zeta, s[SP(j + 128)]);
        s[SP(j + 128)] = s[SP(j)] - t;
        s[SP(j)]       = s[SP(j)] + t;
    }
    __syncthreads();

    /* Level 6: len=64, 2 groups, zeta[2,3] */
    {
        int group = tid >> 6;     /* 0 or 1 */
        int lane  = tid & 0x3F;  /* 0..63 */
        int16_t zeta = ntt_zetas[2 + group];
        int base_idx = group * 128 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 64)]);
        s[SP(base_idx + 64)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 5: len=32, 4 groups, zeta[4..7] */
    {
        int group = tid >> 5;
        int lane  = tid & 0x1F;
        int16_t zeta = ntt_zetas[4 + group];
        int base_idx = group * 64 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 32)]);
        s[SP(base_idx + 32)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 4: len=16, 8 groups, zeta[8..15] */
    {
        int group = tid >> 4;
        int lane  = tid & 0x0F;
        int16_t zeta = ntt_zetas[8 + group];
        int base_idx = group * 32 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 16)]);
        s[SP(base_idx + 16)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 3: len=8, 16 groups, zeta[16..31] */
    {
        int group = tid >> 3;
        int lane  = tid & 0x07;
        int16_t zeta = ntt_zetas[16 + group];
        int base_idx = group * 16 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 8)]);
        s[SP(base_idx + 8)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 2: len=4, 32 groups, zeta[32..63] */
    {
        int group = tid >> 2;
        int lane  = tid & 0x03;
        int16_t zeta = ntt_zetas[32 + group];
        int base_idx = group * 8 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 4)]);
        s[SP(base_idx + 4)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 1: len=2, 64 groups, zeta[64..127] */
    {
        int group = tid >> 1;
        int lane  = tid & 0x01;
        int16_t zeta = ntt_zetas[64 + group];
        int base_idx = group * 4 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 2)]);
        s[SP(base_idx + 2)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

#elif ALGORITHM == ALGO_AIGIS_ENC

    /* Aigis: 8 级, len=128→1, zeta 索引从 1 开始 */
    /* Level 7: len=128, 1 group, zeta[1] */
    {
        int16_t zeta = ntt_zetas[1];
        int j = tid;
        int16_t t = fqmul(zeta, s[SP(j + 128)]);
        s[SP(j + 128)] = s[SP(j)] - t;
        s[SP(j)]       = s[SP(j)] + t;
    }
    __syncthreads();

    /* Level 6: len=64, 2 groups */
    {
        int group = tid >> 6;
        int lane  = tid & 0x3F;
        int16_t zeta = ntt_zetas[2 + group];
        int base_idx = group * 128 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 64)]);
        s[SP(base_idx + 64)] = barrett_reduce((int16_t)(s[SP(base_idx)] - t));
        s[SP(base_idx)]      = barrett_reduce((int16_t)(s[SP(base_idx)] + t));
    }
    __syncthreads();

    /* Level 5: len=32, 4 groups */
    {
        int group = tid >> 5;
        int lane  = tid & 0x1F;
        int16_t zeta = ntt_zetas[4 + group];
        int base_idx = group * 64 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 32)]);
        s[SP(base_idx + 32)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 4: len=16, 8 groups */
    {
        int group = tid >> 4;
        int lane  = tid & 0x0F;
        int16_t zeta = ntt_zetas[8 + group];
        int base_idx = group * 32 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 16)]);
        s[SP(base_idx + 16)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]      = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 3: len=8, 16 groups */
    {
        int group = tid >> 3;
        int lane  = tid & 0x07;
        int16_t zeta = ntt_zetas[16 + group];
        int base_idx = group * 16 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 8)]);
        s[SP(base_idx + 8)] = barrett_reduce((int16_t)(s[SP(base_idx)] - t));
        s[SP(base_idx)]     = barrett_reduce((int16_t)(s[SP(base_idx)] + t));
    }
    __syncthreads();

    /* Level 2: len=4, 32 groups */
    {
        int group = tid >> 2;
        int lane  = tid & 0x03;
        int16_t zeta = ntt_zetas[32 + group];
        int base_idx = group * 8 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 4)]);
        s[SP(base_idx + 4)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 1: len=2, 64 groups */
    {
        int group = tid >> 1;
        int lane  = tid & 0x01;
        int16_t zeta = ntt_zetas[64 + group];
        int base_idx = group * 4 + lane;
        int16_t t = fqmul(zeta, s[SP(base_idx + 2)]);
        s[SP(base_idx + 2)] = s[SP(base_idx)] - t;
        s[SP(base_idx)]     = s[SP(base_idx)] + t;
    }
    __syncthreads();

    /* Level 0: len=1, 128 groups */
    {
        int group = tid;
        int16_t zeta = ntt_zetas[128 + group];
        int base_idx = group * 2;
        int16_t t = fqmul(zeta, s[SP(base_idx + 1)]);
        s[SP(base_idx + 1)] = barrett_reduce((int16_t)(s[SP(base_idx)] - t));
        s[SP(base_idx)]     = barrett_reduce((int16_t)(s[SP(base_idx)] + t));
    }
    __syncthreads();

#endif  /* ALGORITHM for NTT levels */

    /* 写回 */
    base[tid]       = s[SP(tid)];
    base[tid + 128] = s[SP(tid + 128)];
}

/* ================================================================
 *  批量 INVNTT kernel
 * ================================================================ */
__global__ void batch_invntt_kernel(int16_t * __restrict__ polys, int batch_count)
{
    int poly_idx = blockIdx.x;
    if (poly_idx >= batch_count) return;

    int tid = (int)threadIdx.x;

    __shared__ int16_t s[SPAD];

    int16_t *base = polys + poly_idx * PARAM_N;
    s[SP(tid)]       = base[tid];
    s[SP(tid + 128)] = base[tid + 128];
    __syncthreads();

#if ALGORITHM == ALGO_KYBER

    /* Kyber INVNTT: 从 len=2 反向, 使用 +zetas[k--] */
    /* Level 1: len=2 → 64 groups, zeta[64..127] */
    {
        int group = tid >> 1;
        int lane  = tid & 0x01;
        int16_t zeta = ntt_zetas[64 + group];
        int base_idx = group * 4 + lane;
        int16_t t = s[SP(base_idx)];
        s[SP(base_idx)]     = barrett_reduce((int16_t)(t + s[SP(base_idx + 2)]));
        s[SP(base_idx + 2)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 2)] - t));
    }
    __syncthreads();

    /* Level 2: len=4 → 32 groups, zeta[32..63] */
    {
        int group = tid >> 2;
        int lane  = tid & 0x03;
        int16_t zeta = ntt_zetas[32 + group];
        int base_idx = group * 8 + lane;
        int16_t t = s[SP(base_idx)];
        s[SP(base_idx)]     = barrett_reduce((int16_t)(t + s[SP(base_idx + 4)]));
        s[SP(base_idx + 4)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 4)] - t));
    }
    __syncthreads();

    /* Level 3: len=8 → 16 groups, zeta[16..31] */
    {
        int group = tid >> 3;
        int lane  = tid & 0x07;
        int16_t zeta = ntt_zetas[16 + group];
        int base_idx = group * 16 + lane;
        int16_t t = s[SP(base_idx)];
        s[SP(base_idx)]     = barrett_reduce((int16_t)(t + s[SP(base_idx + 8)]));
        s[SP(base_idx + 8)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 8)] - t));
    }
    __syncthreads();

    /* Level 4: len=16 → 8 groups, zeta[8..15] */
    {
        int group = tid >> 4;
        int lane  = tid & 0x0F;
        int16_t zeta = ntt_zetas[8 + group];
        int base_idx = group * 32 + lane;
        int16_t t = s[SP(base_idx)];
        s[SP(base_idx)]      = barrett_reduce((int16_t)(t + s[SP(base_idx + 16)]));
        s[SP(base_idx + 16)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 16)] - t));
    }
    __syncthreads();

    /* Level 5: len=32 → 4 groups, zeta[4..7] */
    {
        int group = tid >> 5;
        int lane  = tid & 0x1F;
        int16_t zeta = ntt_zetas[4 + group];
        int base_idx = group * 64 + lane;
        int16_t t = s[SP(base_idx)];
        s[SP(base_idx)]      = barrett_reduce((int16_t)(t + s[SP(base_idx + 32)]));
        s[SP(base_idx + 32)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 32)] - t));
    }
    __syncthreads();

    /* Level 6: len=64 → 2 groups, zeta[2,3] */
    {
        int group = tid >> 6;
        int lane  = tid & 0x3F;
        int16_t zeta = ntt_zetas[2 + group];
        int base_idx = group * 128 + lane;
        int16_t t = s[SP(base_idx)];
        s[SP(base_idx)]      = barrett_reduce((int16_t)(t + s[SP(base_idx + 64)]));
        s[SP(base_idx + 64)] = fqmul(zeta, (int16_t)(s[SP(base_idx + 64)] - t));
    }
    __syncthreads();

    /* Level 7: len=128 → 1 group, zeta[1] */
    {
        int16_t zeta = ntt_zetas[1];
        int j = tid;
        int16_t t = s[SP(j)];
        s[SP(j)]       = barrett_reduce((int16_t)(t + s[SP(j + 128)]));
        s[SP(j + 128)] = fqmul(zeta, (int16_t)(s[SP(j + 128)] - t));
    }
    __syncthreads();

    /* 归一化 f = 1441 */
    {
        const int16_t f = 1441;
        s[SP(tid)]       = fqmul(s[SP(tid)], f);
        s[SP(tid + 128)] = fqmul(s[SP(tid + 128)], f);
    }
    __syncthreads();

#elif ALGORITHM == ALGO_AIGIS_ENC

    /* Aigis INVNTT: 从 len=1 开始, 使用 ntt_zetas_inv */
    /* Level 0: len=1, 128 groups */
    {
        int group = tid;
        int32_t zeta = ntt_zetas_inv[group];
        int base_idx = group * 2;
        int32_t t = s[SP(base_idx)];
        s[SP(base_idx)]     = (int16_t)(t + s[SP(base_idx + 1)]);
        t -= s[SP(base_idx + 1)];
        s[SP(base_idx + 1)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 1: len=2, 64 groups, Barrett */
    {
        int group = tid >> 1;
        int lane  = tid & 0x01;
        int32_t zeta = ntt_zetas_inv[128 + group];
        int base_idx = group * 4 + lane;
        int32_t t = s[SP(base_idx)];
        s[SP(base_idx)]     = barrett_reduce((int16_t)(t + s[SP(base_idx + 2)]));
        t -= s[SP(base_idx + 2)];
        s[SP(base_idx + 2)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 2: len=4, 32 groups */
    {
        int group = tid >> 2;
        int lane  = tid & 0x03;
        int32_t zeta = ntt_zetas_inv[192 + group];
        int base_idx = group * 8 + lane;
        int32_t t = s[SP(base_idx)];
        s[SP(base_idx)]     = (int16_t)(t + s[SP(base_idx + 4)]);
        t -= s[SP(base_idx + 4)];
        s[SP(base_idx + 4)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 3: len=8, 16 groups, Barrett */
    {
        int group = tid >> 3;
        int lane  = tid & 0x07;
        int32_t zeta = ntt_zetas_inv[224 + group];
        int base_idx = group * 16 + lane;
        int32_t t = s[SP(base_idx)];
        s[SP(base_idx)]     = barrett_reduce((int16_t)(t + s[SP(base_idx + 8)]));
        t -= s[SP(base_idx + 8)];
        s[SP(base_idx + 8)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 4: len=16, 8 groups */
    {
        int group = tid >> 4;
        int lane  = tid & 0x0F;
        int32_t zeta = ntt_zetas_inv[240 + group];
        int base_idx = group * 32 + lane;
        int32_t t = s[SP(base_idx)];
        s[SP(base_idx)]      = (int16_t)(t + s[SP(base_idx + 16)]);
        t -= s[SP(base_idx + 16)];
        s[SP(base_idx + 16)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 5: len=32, 4 groups, Barrett */
    {
        int group = tid >> 5;
        int lane  = tid & 0x1F;
        int32_t zeta = ntt_zetas_inv[248 + group];
        int base_idx = group * 64 + lane;
        int32_t t = s[SP(base_idx)];
        s[SP(base_idx)]      = barrett_reduce((int16_t)(t + s[SP(base_idx + 32)]));
        t -= s[SP(base_idx + 32)];
        s[SP(base_idx + 32)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 6: len=64, 2 groups */
    {
        int group = tid >> 6;
        int lane  = tid & 0x3F;
        int32_t zeta = ntt_zetas_inv[252 + group];
        int base_idx = group * 128 + lane;
        int32_t t = s[SP(base_idx)];
        s[SP(base_idx)]      = (int16_t)(t + s[SP(base_idx + 64)]);
        t -= s[SP(base_idx + 64)];
        s[SP(base_idx + 64)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

    /* Level 7: len=128, 1 group, 含 N^{-1} 归一化 */
    {
        int32_t zeta = ntt_zetas_inv[254];
        int j = tid;
        int32_t t = s[SP(j)];
        /* r[j] = (r[j] + r[j+128]) * N^{-1} mod Q */
        s[SP(j)]       = montgomery_reduce(256 * (t + s[SP(j + 128)]));
        t -= s[SP(j + 128)];
        s[SP(j + 128)] = montgomery_reduce((int32_t)zeta * (int16_t)t);
    }
    __syncthreads();

#endif  /* ALGORITHM for INVNTT */

    base[tid]       = s[SP(tid)];
    base[tid + 128] = s[SP(tid + 128)];
}

/* ================================================================
 *  Host 启动封装
 * ================================================================ */

static inline void launch_batch_ntt(int16_t *d_polys, int batch_count,
                                     cudaStream_t stream = 0)
{
    batch_ntt_kernel<<<batch_count, 128, 0, stream>>>(d_polys, batch_count);
}

static inline void launch_batch_invntt(int16_t *d_polys, int batch_count,
                                        cudaStream_t stream = 0)
{
    batch_invntt_kernel<<<batch_count, 128, 0, stream>>>(d_polys, batch_count);
}

#undef SP
#undef SPAD

#endif /* BATCH_NTT_CUH */
