#include "hip/hip_runtime.h"
/*
 * batch_keygen.cuh — 分解式批量密钥生成 pipeline
 *
 * 核心优化:
 *   1. 采样 (SHAKE-heavy) 使用 1 warp/实例 (32 线程并行生成所有多项式)
 *   2. NTT 使用 shared-memory 批量 kernel (128 线程/poly)
 *   3. 矩阵向量乘使用 2D grid (batch × K), 每系数一线程
 *   4. 元素级运算使用 256 线程/block 的批量 kernel
 *   5. 打包由 32 线程/block 独立执行
 *
 * Pipeline:
 *   [1] 采样: seed → A, s1, s2          (warp 级并行, #if 算法分叉)
 *   [2] copy s1 → s1hat
 *   [3] NTT(s1hat)                       (shared-mem batch)
 *   [4] t = A · s1hat                    (2D grid matvec)
 *   [5] reduce + INVNTT(t)              (batch kernels)
 *   [6] t += s2                          (batch add)
 *   [7] 打包 pk, sk                      (#if 算法分叉)
 */

#ifndef BATCH_KEYGEN_CUH
#define BATCH_KEYGEN_CUH

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>
#include "params.h"
#include "reduce.cuh"
#include "ntt.cuh"
#include "fips202.cuh"
#include "poly.cuh"
#include "polyvec.cuh"
#include "packing.cuh"
#include "batch_ntt.cuh"
#include "batch_ops.cuh"
#include "sign.cuh"
#include "symmetric.cuh"

#ifndef BATCH_KEYGEN_SAMPLE_SPLIT_FAST
#define BATCH_KEYGEN_SAMPLE_SPLIT_FAST 0
#endif

#ifndef BATCH_KEYGEN_MATRIX_A_FAST
#define BATCH_KEYGEN_MATRIX_A_FAST BATCH_KEYGEN_SAMPLE_SPLIT_FAST
#endif

#ifndef BATCH_KEYGEN_SECRET_ETA_FAST
#define BATCH_KEYGEN_SECRET_ETA_FAST BATCH_KEYGEN_SAMPLE_SPLIT_FAST
#endif

#ifndef BATCH_KEYGEN_MATRIX_A_COOP
#define BATCH_KEYGEN_MATRIX_A_COOP 0
#endif

#ifndef BATCH_KEYGEN_MATRIX_A_LANEOPT
#define BATCH_KEYGEN_MATRIX_A_LANEOPT 0
#endif

#ifndef BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP
#define BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP 0
#endif

#ifndef BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP_LANES
#define BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP_LANES 16
#endif

#ifndef BATCH_KEYGEN_SECRET_ETA_COOP
#define BATCH_KEYGEN_SECRET_ETA_COOP 0
#endif

#ifndef BATCH_KEYGEN_SECRET_ETA_COOP_LANES
#define BATCH_KEYGEN_SECRET_ETA_COOP_LANES 16
#endif

#ifndef BATCH_KEYGEN_SECRET_ETA_AIGIS5_SPLIT
#define BATCH_KEYGEN_SECRET_ETA_AIGIS5_SPLIT 0
#endif

/* ================================================================
 * 缓冲区结构体 — 两种算法共用
 * ================================================================ */
struct BatchKeygenBuffers {
    coeff_t *d_mat;       /* batch * K * L * N */
    coeff_t *d_s1;        /* batch * L * N */
    coeff_t *d_s1hat;     /* batch * L * N (NTT domain) */
    coeff_t *d_s2;        /* batch * K * N */
    coeff_t *d_t;         /* batch * K * N */
    coeff_t *d_t1;        /* batch * K * N — power2round high bits */
    coeff_t *d_t0;        /* batch * K * N — power2round low bits */
    coeff_t *d_t1_hat;    /* batch * K * N — NTT(t1 << D), verify material */
    coeff_t *d_s2_ntt;    /* batch * K * N — NTT(s2), sign material */
    coeff_t *d_t0_ntt;    /* batch * K * N — NTT(t0), sign material */
    unsigned char *d_tr;  /* batch * TRBYTES — H(pk) */
    unsigned char *d_pks; /* batch * CRYPTO_PUBLICKEYBYTES */
    unsigned char *d_sks; /* batch * CRYPTO_SECRETKEYBYTES */
    unsigned char *d_buf; /* per-instance 辅助数据 (rho, key 等) */
    int max_batch;
};

typedef struct {
    float sample_ms;
    float seed_expand_ms;
    float matrix_a_sample_ms;
    float secret_eta_sample_ms;
    float sample_launch_gap_ms;
    float matrix_a_coop_ms;
    float secret_eta_coop_ms;
    float copy_ms;
    float ntt_ms;
    float matvec_ms;
    float post_ms;
    float p2r_ms;
    float pack_ms;
    float pack_inner_ms;
    float pack_fused_ms;
    float pack_body_ms;
    float pack_header_ms;
    float pack_t1_ms;
    float pack_eta_ms;
    float pack_t0_ms;
    float tr_hash_ms;
    float shared_a_ms;
    float material_ms;
    int matrix_a_coop_lanes;
    int secret_eta_coop_lanes;
} KeygenProfile;

typedef struct {
    float old_fused_ms;
    float shared_a_ms;
    float split_seed_ms;
    float split_matrix_a_ms;
    float split_eta_ms;
    float split_total_ms;
    float split_launch_gap_ms;
    float split_matrix_a_coop_ms;
    float split_eta_coop_ms;
    int split_matrix_a_coop_lanes;
    int split_eta_coop_lanes;
} KeygenSampleOnlyProfile;

typedef enum {
    KEYGEN_COMPARE_STAGE_NONE = 0,
    KEYGEN_COMPARE_STAGE_BUF,
    KEYGEN_COMPARE_STAGE_MAT,
    KEYGEN_COMPARE_STAGE_S1,
    KEYGEN_COMPARE_STAGE_S2,
    KEYGEN_COMPARE_STAGE_S1HAT_COPY,
    KEYGEN_COMPARE_STAGE_S1HAT_NTT,
    KEYGEN_COMPARE_STAGE_T_MATVEC,
    KEYGEN_COMPARE_STAGE_T,
    KEYGEN_COMPARE_STAGE_T1,
    KEYGEN_COMPARE_STAGE_T0,
    KEYGEN_COMPARE_STAGE_PK,
    KEYGEN_COMPARE_STAGE_SK,
    KEYGEN_COMPARE_STAGE_TR,
} KeygenCompareStage;

typedef struct {
    KeygenCompareStage stage;
    int instance;
    size_t byte_offset;
    size_t element_offset;
    int64_t ref_value;
    int64_t cand_value;
} KeygenCompareResult;

static inline void keygen_profile_clear(KeygenProfile *p) {
    if (p) memset(p, 0, sizeof(*p));
}

static inline void keygen_sample_only_profile_clear(KeygenSampleOnlyProfile *p) {
    if (p) memset(p, 0, sizeof(*p));
}

static inline void keygen_compare_result_clear(KeygenCompareResult *r) {
    if (r) memset(r, 0, sizeof(*r));
}

static inline const char *keygen_compare_stage_name(KeygenCompareStage stage) {
    switch (stage) {
    case KEYGEN_COMPARE_STAGE_BUF: return "d_buf";
    case KEYGEN_COMPARE_STAGE_MAT: return "d_mat";
    case KEYGEN_COMPARE_STAGE_S1: return "d_s1";
    case KEYGEN_COMPARE_STAGE_S2: return "d_s2";
    case KEYGEN_COMPARE_STAGE_S1HAT_COPY: return "d_s1hat-copy";
    case KEYGEN_COMPARE_STAGE_S1HAT_NTT: return "d_s1hat-ntt";
    case KEYGEN_COMPARE_STAGE_T_MATVEC: return "d_t-matvec";
    case KEYGEN_COMPARE_STAGE_T: return "d_t";
    case KEYGEN_COMPARE_STAGE_T1: return "d_t1";
    case KEYGEN_COMPARE_STAGE_T0: return "d_t0";
    case KEYGEN_COMPARE_STAGE_PK: return "pk";
    case KEYGEN_COMPARE_STAGE_SK: return "sk";
    case KEYGEN_COMPARE_STAGE_TR: return "tr";
    default: return "none";
    }
}

static inline void keygen_profile_add(float *dst, hipEvent_t a, hipEvent_t b) {
    float ms = 0.0f;
    hipEventElapsedTime(&ms, a, b);
    *dst += ms;
}

static inline void keygen_profile_finalize_sample(
    KeygenProfile *p,
    float component_ms)
{
    if (!p) return;
    const float gap = p->sample_ms - component_ms;
    p->sample_launch_gap_ms = gap > 0.0f ? gap : 0.0f;
}

/* ================================================================
 * 算子级并行采样 kernel — 仿照「合并的第五版」warp-cooperative 思路
 *
 * 设计: 1 warp (32 线程) 处理 1 个 instance
 *   lane 0: 派生 seed (SHAKE256 展开)
 *   所有 32 lanes: 并行生成矩阵 A + s1 + s2 的所有多项式
 *   总共 PARAM_K*PARAM_L + PARAM_L + PARAM_K 个多项式各自独立 SHAKE 流
 *   每个 lane 处理 p = lane, lane+32, lane+64, ... 的多项式
 *   多项式各自独立 → 零同步, 32× 并行化 SHAKE 调用
 *
 * 性能: 采样阶段从 O(K*L+L+K) 串行降到 O(ceil((K*L+L+K)/32))
 * ================================================================ */
#define WP_KG_WARP_SIZE    32
#define WP_KG_WARPS_BLOCK  4
#define WP_KG_TPB          (WP_KG_WARP_SIZE * WP_KG_WARPS_BLOCK)
/* 共享种子缓冲区大小 (per warp): rho + rhoprime/eta_seed + key */
#define WP_KG_SEED_BYTES   (2 * SEEDBYTES + CRHBYTES)
#define WP_KG_MAX_SUBWARPS_PER_BLOCK (WP_KG_TPB / 8)
#define WP_KG_MATRIX_COOP_BUF_BYTES \
    (POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES + STREAM128_BLOCKBYTES + 2)
#define WP_KG_ETA_COOP_BUF_BYTES \
    (POLY_UNIFORM_ETA2_NBLOCKS * STREAM256_BLOCKBYTES + STREAM256_BLOCKBYTES)

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_warp_sample_kernel(
    coeff_t * __restrict__ d_mat,
    coeff_t * __restrict__ d_s1,
    coeff_t * __restrict__ d_s2,
    unsigned char * __restrict__ d_buf,
    const unsigned char * __restrict__ d_base_seed,
    int batch_count)
{
    __shared__ unsigned char sh_seeds[WP_KG_WARPS_BLOCK][WP_KG_SEED_BYTES];

    const int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) / WP_KG_WARP_SIZE;
    const int lane   = threadIdx.x & (WP_KG_WARP_SIZE - 1);
    const int warp_l = threadIdx.x / WP_KG_WARP_SIZE;

    if (warp_g >= batch_count) return;

    unsigned char *my_seeds = sh_seeds[warp_l];

    /* lane 0: 派生 per-instance seed 并 SHAKE256 展开到 shared memory */
    if (lane == 0) {
        uint8_t seed_in[SEEDBYTES];
        for (int i = 0; i < SEEDBYTES; i++) seed_in[i] = d_base_seed[i];
        seed_in[SEEDBYTES - 4] ^= (uint8_t)(warp_g);
        seed_in[SEEDBYTES - 3] ^= (uint8_t)(warp_g >> 8);
        seed_in[SEEDBYTES - 2] ^= (uint8_t)(warp_g >> 16);
        seed_in[SEEDBYTES - 1] ^= (uint8_t)(warp_g >> 24);

#if ALGORITHM == ALGO_MLDSA
        /* ML-DSA: H(seed || K || L) → rho(32) | rhoprime(64) | key(32) */
        uint8_t buf[2 * SEEDBYTES + CRHBYTES];
        for (int i = 0; i < SEEDBYTES; i++) buf[i] = seed_in[i];
        buf[SEEDBYTES]     = PARAM_K;
        buf[SEEDBYTES + 1] = PARAM_L;
        shake256(buf, 2 * SEEDBYTES + CRHBYTES, buf, SEEDBYTES + 2);
        for (int i = 0; i < 2 * SEEDBYTES + CRHBYTES; i++) my_seeds[i] = buf[i];
#elif ALGORITHM == ALGO_AIGIS
        /* Aigis: H(seed) → eta_seed(32) | rho(32) | key(32) */
        uint8_t buf[3 * SEEDBYTES];
        shake256(buf, 3 * SEEDBYTES, seed_in, SEEDBYTES);
        for (int i = 0; i < 3 * SEEDBYTES; i++) my_seeds[i] = buf[i];
#endif

        /* 存 rho, key 到 d_buf (供后续 pack kernel 使用) */
        unsigned char *my_buf = d_buf + (size_t)warp_g * (2 * SEEDBYTES + CRHBYTES);
#if ALGORITHM == ALGO_MLDSA
        const uint8_t *rho = my_seeds;
        const uint8_t *key = my_seeds + SEEDBYTES + CRHBYTES;
        const uint8_t *rhp = my_seeds + SEEDBYTES;
        for (int i = 0; i < SEEDBYTES; i++) my_buf[i] = rho[i];
        for (int i = 0; i < SEEDBYTES; i++) my_buf[SEEDBYTES + i] = key[i];
        for (int i = 0; i < CRHBYTES; i++) my_buf[2 * SEEDBYTES + i] = rhp[i];
#elif ALGORITHM == ALGO_AIGIS
    const uint8_t *eta_seed = my_seeds;
        const uint8_t *rho = my_seeds + SEEDBYTES;
        const uint8_t *key = my_seeds + 2 * SEEDBYTES;
        for (int i = 0; i < SEEDBYTES; i++) my_buf[i] = rho[i];
        for (int i = 0; i < SEEDBYTES; i++) my_buf[SEEDBYTES + i] = key[i];
    for (int i = 0; i < SEEDBYTES; i++) my_buf[2 * SEEDBYTES + i] = eta_seed[i];
#endif
    }
    __syncwarp();

    /* 所有 32 lanes 并行生成多项式 */
    /* 多项式索引分配:
     *   p = 0 .. K*L-1        → A[p/L][p%L]
     *   p = K*L .. K*L+L-1   → s1[p - K*L]
     *   p = K*L+L .. end      → s2[p - K*L - L]
     */
    const int TOTAL_POLYS = PARAM_K * PARAM_L + PARAM_L + PARAM_K;

#if ALGORITHM == ALGO_MLDSA
    const uint8_t *rho      = my_seeds;
    const uint8_t *rhoprime = my_seeds + SEEDBYTES;
#elif ALGORITHM == ALGO_AIGIS
    const uint8_t *eta_seed = my_seeds;
    const uint8_t *rho      = my_seeds + SEEDBYTES;
#endif

    for (int p = lane; p < TOTAL_POLYS; p += WP_KG_WARP_SIZE) {
        coeff_t *dst;

        if (p < PARAM_K * PARAM_L) {
            /* 矩阵 A[row][col] */
            int row = p / PARAM_L;
            int col = p % PARAM_L;
            dst = d_mat + (size_t)warp_g * PARAM_K * PARAM_L * PARAM_N + p * PARAM_N;
            poly_uniform_to(dst, rho, MATRIX_NONCE(row, col));
        } else if (p < PARAM_K * PARAM_L + PARAM_L) {
            /* 秘密向量 s1[j] */
            int j = p - PARAM_K * PARAM_L;
#if ALGORITHM == ALGO_MLDSA
            dst = d_s1 + (size_t)warp_g * PARAM_L * PARAM_N + (size_t)j * PARAM_N;
            poly_uniform_eta_s1_to(dst, rhoprime, j);
#elif ALGORITHM == ALGO_AIGIS
            dst = d_s1 + (size_t)warp_g * PARAM_L * PARAM_N + (size_t)j * PARAM_N;
            poly_uniform_eta_s1_to(dst, eta_seed, (uint16_t)j);
#endif
        } else {
            /* 秘密向量 s2[k] */
            int k = p - PARAM_K * PARAM_L - PARAM_L;
#if ALGORITHM == ALGO_MLDSA
            dst = d_s2 + (size_t)warp_g * PARAM_K * PARAM_N + (size_t)k * PARAM_N;
            poly_uniform_eta_s2_to(dst, rhoprime, PARAM_L + k);
#elif ALGORITHM == ALGO_AIGIS
            dst = d_s2 + (size_t)warp_g * PARAM_K * PARAM_N + (size_t)k * PARAM_N;
            poly_uniform_eta_s2_to(dst, eta_seed, (uint16_t)(PARAM_L + k));
#endif
        }
    }
}

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_seed_expand_kernel(
    unsigned char * __restrict__ d_buf,
    const unsigned char * __restrict__ d_base_seed,
    int batch_count)
{
    const int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) / WP_KG_WARP_SIZE;
    const int lane   = threadIdx.x & (WP_KG_WARP_SIZE - 1);

    if (warp_g >= batch_count || lane != 0) return;

    unsigned char *my_buf = d_buf + (size_t)warp_g * (2 * SEEDBYTES + CRHBYTES);
    uint8_t seed_in[SEEDBYTES];
    for (int i = 0; i < SEEDBYTES; i++) seed_in[i] = d_base_seed[i];
    seed_in[SEEDBYTES - 4] ^= (uint8_t)(warp_g);
    seed_in[SEEDBYTES - 3] ^= (uint8_t)(warp_g >> 8);
    seed_in[SEEDBYTES - 2] ^= (uint8_t)(warp_g >> 16);
    seed_in[SEEDBYTES - 1] ^= (uint8_t)(warp_g >> 24);

#if ALGORITHM == ALGO_MLDSA
    uint8_t buf[2 * SEEDBYTES + CRHBYTES];
    for (int i = 0; i < SEEDBYTES; i++) buf[i] = seed_in[i];
    buf[SEEDBYTES]     = PARAM_K;
    buf[SEEDBYTES + 1] = PARAM_L;
    shake256(buf, 2 * SEEDBYTES + CRHBYTES, buf, SEEDBYTES + 2);

    const uint8_t *rho = buf;
    const uint8_t *rhoprime = buf + SEEDBYTES;
    const uint8_t *key = buf + SEEDBYTES + CRHBYTES;
    for (int i = 0; i < SEEDBYTES; i++) my_buf[i] = rho[i];
    for (int i = 0; i < SEEDBYTES; i++) my_buf[SEEDBYTES + i] = key[i];
    for (int i = 0; i < CRHBYTES; i++) my_buf[2 * SEEDBYTES + i] = rhoprime[i];
#elif ALGORITHM == ALGO_AIGIS
    uint8_t buf[3 * SEEDBYTES];
    shake256(buf, 3 * SEEDBYTES, seed_in, SEEDBYTES);

    const uint8_t *eta_seed = buf;
    const uint8_t *rho = buf + SEEDBYTES;
    const uint8_t *key = buf + 2 * SEEDBYTES;
    for (int i = 0; i < SEEDBYTES; i++) my_buf[i] = rho[i];
    for (int i = 0; i < SEEDBYTES; i++) my_buf[SEEDBYTES + i] = key[i];
    for (int i = 0; i < SEEDBYTES; i++) my_buf[2 * SEEDBYTES + i] = eta_seed[i];
#endif
}

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_matrix_a_sample_kernel(
    coeff_t * __restrict__ d_mat,
    const unsigned char * __restrict__ d_buf,
    int batch_count)
{
    const int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) / WP_KG_WARP_SIZE;
    const int lane   = threadIdx.x & (WP_KG_WARP_SIZE - 1);

    if (warp_g >= batch_count) return;

    const unsigned char *my_buf = d_buf + (size_t)warp_g * (2 * SEEDBYTES + CRHBYTES);
    const uint8_t *rho = my_buf;
    const int total = PARAM_K * PARAM_L;

    for (int p = lane; p < total; p += WP_KG_WARP_SIZE) {
        int row = p / PARAM_L;
        int col = p % PARAM_L;
        coeff_t *dst = d_mat + (size_t)warp_g * PARAM_K * PARAM_L * PARAM_N + (size_t)p * PARAM_N;
        poly_uniform_to(dst, rho, MATRIX_NONCE(row, col));
    }
}

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_matrix_a_laneopt_kernel(
    coeff_t * __restrict__ d_mat,
    const unsigned char * __restrict__ d_buf,
    int batch_count)
{
    __shared__ uint8_t sh_rho[WP_KG_WARPS_BLOCK][SEEDBYTES];

    const int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) / WP_KG_WARP_SIZE;
    const int lane   = threadIdx.x & (WP_KG_WARP_SIZE - 1);
    const int warp_l = threadIdx.x / WP_KG_WARP_SIZE;

    if (warp_g >= batch_count) return;

    const unsigned char *my_buf = d_buf + (size_t)warp_g * (2 * SEEDBYTES + CRHBYTES);
    uint8_t *rho_local = sh_rho[warp_l];
    if (lane < SEEDBYTES)
        rho_local[lane] = my_buf[lane];
    __syncwarp();

    const size_t inst_mat_off = (size_t)warp_g * PARAM_K * PARAM_L * PARAM_N;
    const int total = PARAM_K * PARAM_L;
    for (int p = lane; p < total; p += WP_KG_WARP_SIZE) {
        const int row = p / PARAM_L;
        const int col = p % PARAM_L;
        coeff_t *dst = d_mat + inst_mat_off + (size_t)p * PARAM_N;
        poly_uniform_to(dst, rho_local, MATRIX_NONCE(row, col));
    }
}

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_secret_sample_kernel(
    coeff_t * __restrict__ d_s1,
    coeff_t * __restrict__ d_s2,
    const unsigned char * __restrict__ d_buf,
    int batch_count)
{
    const int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) / WP_KG_WARP_SIZE;
    const int lane   = threadIdx.x & (WP_KG_WARP_SIZE - 1);

    if (warp_g >= batch_count) return;

    const unsigned char *my_buf = d_buf + (size_t)warp_g * (2 * SEEDBYTES + CRHBYTES);
    const int total = PARAM_L + PARAM_K;

#if ALGORITHM == ALGO_MLDSA
    const uint8_t *rhoprime = my_buf + 2 * SEEDBYTES;
#elif ALGORITHM == ALGO_AIGIS
    const uint8_t *eta_seed = my_buf + 2 * SEEDBYTES;
#endif

    for (int p = lane; p < total; p += WP_KG_WARP_SIZE) {
        if (p < PARAM_L) {
            int j = p;
            coeff_t *dst = d_s1 + (size_t)warp_g * PARAM_L * PARAM_N + (size_t)j * PARAM_N;
#if ALGORITHM == ALGO_MLDSA
            poly_uniform_eta_s1_to(dst, rhoprime, j);
#elif ALGORITHM == ALGO_AIGIS
            poly_uniform_eta_s1_to(dst, eta_seed, (uint16_t)j);
#endif
        } else {
            int k = p - PARAM_L;
            coeff_t *dst = d_s2 + (size_t)warp_g * PARAM_K * PARAM_N + (size_t)k * PARAM_N;
#if ALGORITHM == ALGO_MLDSA
            poly_uniform_eta_s2_to(dst, rhoprime, PARAM_L + k);
#elif ALGORITHM == ALGO_AIGIS
            poly_uniform_eta_s2_to(dst, eta_seed, (uint16_t)(PARAM_L + k));
#endif
        }
    }
}

template<int SUBWARP_LANES>
__device__ __forceinline__ unsigned long long wp_kg_subwarp_mask(int lane_in_warp)
{
    const int base = lane_in_warp - (lane_in_warp & (SUBWARP_LANES - 1));
    return (0xFFFFFFFFull >> (32 - SUBWARP_LANES)) << base;
}

template<int SUBWARP_LANES>
__device__ __forceinline__ int wp_kg_subwarp_exclusive_scan(int value,
                                                            unsigned long long mask,
                                                            int sublane)
{
    int scan = value;
#pragma unroll
    for (int offset = 1; offset < SUBWARP_LANES; offset <<= 1) {
        int other = __shfl_up_sync(mask, scan, offset);
        if (sublane >= offset)
            scan += other;
    }
    return scan - value;
}

template<int SUBWARP_LANES>
__device__ __forceinline__ int wp_kg_subwarp_sum(int value,
                                                 unsigned long long mask,
                                                 int leader_lane)
{
    int sum = value;
#pragma unroll
    for (int offset = SUBWARP_LANES >> 1; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(mask, sum, offset);
    return __shfl_sync(mask, sum, leader_lane);
}

__device__ __forceinline__ void wp_kg_store_coeff(coeff_t *dst,
                                                  coeff_t *dst_copy,
                                                  int idx,
                                                  coeff_t value)
{
    dst[idx] = value;
    if (dst_copy)
        dst_copy[idx] = value;
}

template<int SUBWARP_LANES>
__device__ void wp_kg_uniform_coop_sample_to(
    coeff_t *dst,
    const uint8_t *seed,
    uint16_t nonce,
    uint8_t *buf,
    int *ctr_ptr,
    unsigned int *buflen_ptr,
    unsigned long long mask,
    int sublane,
    int leader_lane)
{
    stream128_state state;

    if (sublane == 0) {
#if ALGORITHM == ALGO_MLDSA
        stream128_init(&state, seed, nonce);
#elif ALGORITHM == ALGO_AIGIS
        aigis_shake128_stream_init(&state, seed, (uint8_t)nonce);
#endif
        stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);
        *ctr_ptr = 0;
        *buflen_ptr = POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES;
    }
    __syncwarp(mask);

    while (1) {
        int cur_ctr = *ctr_ptr;
        if (cur_ctr >= PARAM_N)
            break;

        const unsigned int buflen = *buflen_ptr;
        const int total_candidates = (int)(buflen / 3u);

        for (int base = 0; base < total_candidates; base += SUBWARP_LANES) {
            cur_ctr = *ctr_ptr;
            if (cur_ctr >= PARAM_N)
                break;

            const int cand = base + sublane;
            int accept = 0;
            coeff_t value = 0;

            if (cand < total_candidates) {
                const size_t pos = (size_t)cand * 3u;
                uint32_t t = buf[pos]
                           | ((uint32_t)buf[pos + 1] << 8)
                           | ((uint32_t)buf[pos + 2] << 16);
                t &= (1u << PARAM_QBITS) - 1u;
                if (t < (uint32_t)PARAM_Q) {
                    accept = 1;
                    value = (coeff_t)t;
                }
            }

            const int prefix = wp_kg_subwarp_exclusive_scan<SUBWARP_LANES>(accept, mask, sublane);
            const int accepted = wp_kg_subwarp_sum<SUBWARP_LANES>(accept, mask, leader_lane);

            if (accept) {
                const int out_idx = cur_ctr + prefix;
                if (out_idx < PARAM_N)
                    dst[out_idx] = value;
            }

            if (sublane == 0) {
                const int next_ctr = cur_ctr + accepted;
                *ctr_ptr = next_ctr < PARAM_N ? next_ctr : PARAM_N;
            }
            __syncwarp(mask);
        }

        cur_ctr = *ctr_ptr;
        if (cur_ctr >= PARAM_N)
            break;

        if (sublane == 0) {
            const unsigned int buflen_local = *buflen_ptr;
            const unsigned int off = buflen_local % 3u;
            for (unsigned int i = 0; i < off; ++i)
                buf[i] = buf[buflen_local - off + i];
            stream128_squeezeblocks(buf + off, 1, &state);
            *buflen_ptr = STREAM128_BLOCKBYTES + off;
        }
        __syncwarp(mask);
    }
}

template<int SUBWARP_LANES>
__device__ void wp_kg_eta_mldsa_coop_sample_to(
    coeff_t *dst,
    coeff_t *dst_copy,
    const uint8_t *seed,
    uint16_t nonce,
    int eta,
    int init_blocks,
    uint8_t *buf,
    int *ctr_ptr,
    unsigned int *buflen_ptr,
    unsigned long long mask,
    int sublane,
    int leader_lane)
{
    stream256_state state;

    if (sublane == 0) {
        stream256_init(&state, seed, nonce);
        stream256_squeezeblocks(buf, init_blocks, &state);
        *ctr_ptr = 0;
        *buflen_ptr = (unsigned int)init_blocks * STREAM256_BLOCKBYTES;
    }
    __syncwarp(mask);

    while (1) {
        int cur_ctr = *ctr_ptr;
        if (cur_ctr >= PARAM_N)
            break;

        const int total_bytes = (int)(*buflen_ptr);
        for (int base = 0; base < total_bytes; base += SUBWARP_LANES) {
            cur_ctr = *ctr_ptr;
            if (cur_ctr >= PARAM_N)
                break;

            const int byte_pos = base + sublane;
            int have0 = 0, have1 = 0;
            coeff_t value0 = 0, value1 = 0;
            int count = 0;

            if (byte_pos < total_bytes) {
                uint32_t t0 = buf[byte_pos] & 0x0F;
                uint32_t t1 = buf[byte_pos] >> 4;
                if (eta == 2) {
                    if (t0 < 15) {
                        t0 = t0 - ((205 * t0) >> 10) * 5;
                        value0 = 2 - (int32_t)t0;
                        have0 = 1;
                        count++;
                    }
                    if (t1 < 15) {
                        t1 = t1 - ((205 * t1) >> 10) * 5;
                        value1 = 2 - (int32_t)t1;
                        have1 = 1;
                        count++;
                    }
                } else {
                    if (t0 < 9) {
                        value0 = 4 - (int32_t)t0;
                        have0 = 1;
                        count++;
                    }
                    if (t1 < 9) {
                        value1 = 4 - (int32_t)t1;
                        have1 = 1;
                        count++;
                    }
                }
            }

            const int prefix = wp_kg_subwarp_exclusive_scan<SUBWARP_LANES>(count, mask, sublane);
            const int accepted = wp_kg_subwarp_sum<SUBWARP_LANES>(count, mask, leader_lane);

            if (have0) {
                const int out_idx = cur_ctr + prefix;
                if (out_idx < PARAM_N)
                    wp_kg_store_coeff(dst, dst_copy, out_idx, value0);
            }
            if (have1) {
                const int out_idx = cur_ctr + prefix + have0;
                if (out_idx < PARAM_N)
                    wp_kg_store_coeff(dst, dst_copy, out_idx, value1);
            }

            if (sublane == 0) {
                const int next_ctr = cur_ctr + accepted;
                *ctr_ptr = next_ctr < PARAM_N ? next_ctr : PARAM_N;
            }
            __syncwarp(mask);
        }

        cur_ctr = *ctr_ptr;
        if (cur_ctr >= PARAM_N)
            break;

        if (sublane == 0) {
            stream256_squeezeblocks(buf, 1, &state);
            *buflen_ptr = STREAM256_BLOCKBYTES;
        }
        __syncwarp(mask);
    }
}

template<int SUBWARP_LANES>
__device__ void wp_kg_eta1_aigis_coop_sample_to(
    coeff_t *dst,
    coeff_t *dst_copy,
    const uint8_t *seed,
    uint16_t nonce,
    uint8_t *buf,
    int *ctr_ptr,
    unsigned int *buflen_ptr,
    unsigned long long mask,
    int sublane,
    int leader_lane)
{
    stream256_state state;

    if (sublane == 0) {
        aigis_shake256_eta_init(&state, seed, (uint8_t)nonce);
        stream256_squeezeblocks(buf, POLY_UNIFORM_ETA1_NBLOCKS, &state);
        *ctr_ptr = 0;
        *buflen_ptr = POLY_UNIFORM_ETA1_NBLOCKS * STREAM256_BLOCKBYTES;
    }
    __syncwarp(mask);

    while (1) {
        int cur_ctr = *ctr_ptr;
        if (cur_ctr >= PARAM_N)
            break;

#if PARAM_ETA_S1 == 1
        const int total_units = (int)(*buflen_ptr);
#else
        const int total_units = (int)(*buflen_ptr / 3u);
#endif

        for (int base = 0; base < total_units; base += SUBWARP_LANES) {
            cur_ctr = *ctr_ptr;
            if (cur_ctr >= PARAM_N)
                break;

            const int unit = base + sublane;
            coeff_t values[8];
            int count = 0;

#if PARAM_ETA_S1 == 1
            if (unit < total_units) {
                const uint32_t byte = buf[unit];
                const uint32_t t0 = byte & 0x03;
                const uint32_t t1 = (byte >> 2) & 0x03;
                const uint32_t t2 = (byte >> 4) & 0x03;
                const uint32_t t3 = byte >> 6;
                if (t0 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t0;
                if (t1 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t1;
                if (t2 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t2;
                if (t3 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t3;
            }
#else
            if (unit < total_units) {
                const int pos = unit * 3;
                const uint32_t t0 = buf[pos] & 0x07;
                const uint32_t t1 = (buf[pos] >> 3) & 0x07;
                const uint32_t t2 = (buf[pos] >> 6) | ((uint32_t)(buf[pos + 1] & 0x01) << 2);
                const uint32_t t3 = (buf[pos + 1] >> 1) & 0x07;
                const uint32_t t4 = (buf[pos + 1] >> 4) & 0x07;
                const uint32_t t5 = (buf[pos + 1] >> 7) | ((uint32_t)(buf[pos + 2] & 0x03) << 1);
                const uint32_t t6 = (buf[pos + 2] >> 2) & 0x07;
                const uint32_t t7 = buf[pos + 2] >> 5;
                if (t0 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t0;
                if (t1 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t1;
                if (t2 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t2;
                if (t3 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t3;
                if (t4 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t4;
                if (t5 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t5;
                if (t6 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t6;
                if (t7 <= 2u * PARAM_ETA_S1) values[count++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t7;
            }
#endif

            const int prefix = wp_kg_subwarp_exclusive_scan<SUBWARP_LANES>(count, mask, sublane);
            const int accepted = wp_kg_subwarp_sum<SUBWARP_LANES>(count, mask, leader_lane);

            for (int i = 0; i < count; ++i) {
                const int out_idx = cur_ctr + prefix + i;
                if (out_idx < PARAM_N)
                    wp_kg_store_coeff(dst, dst_copy, out_idx, values[i]);
            }

            if (sublane == 0) {
                const int next_ctr = cur_ctr + accepted;
                *ctr_ptr = next_ctr < PARAM_N ? next_ctr : PARAM_N;
            }
            __syncwarp(mask);
        }

        cur_ctr = *ctr_ptr;
        if (cur_ctr >= PARAM_N)
            break;

        if (sublane == 0) {
            stream256_squeezeblocks(buf, 1, &state);
            *buflen_ptr = STREAM256_BLOCKBYTES;
        }
        __syncwarp(mask);
    }
}

template<int SUBWARP_LANES>
__device__ void wp_kg_eta2_aigis_coop_sample_to(
    coeff_t *dst,
    const uint8_t *seed,
    uint16_t nonce,
    uint8_t *buf,
    int *ctr_ptr,
    unsigned int *buflen_ptr,
    unsigned long long mask,
    int sublane,
    int leader_lane)
{
#if PARAM_ETA_S2 == 5
    if (sublane == 0)
        poly_uniform_eta_s2_to(dst, seed, nonce);
    __syncwarp(mask);
#else
    stream256_state state;

    if (sublane == 0) {
        aigis_shake256_eta_init(&state, seed, (uint8_t)nonce);
        stream256_squeezeblocks(buf, 2, &state);
        *ctr_ptr = 0;
        *buflen_ptr = 2 * STREAM256_BLOCKBYTES;
    }
    __syncwarp(mask);

    while (1) {
        int cur_ctr = *ctr_ptr;
        if (cur_ctr >= PARAM_N)
            break;

        const int total_bytes = (int)(*buflen_ptr);
        for (int base = 0; base < total_bytes; base += SUBWARP_LANES) {
            cur_ctr = *ctr_ptr;
            if (cur_ctr >= PARAM_N)
                break;

            const int byte_pos = base + sublane;
            int have0 = 0, have1 = 0;
            coeff_t value0 = 0, value1 = 0;
            int count = 0;

            if (byte_pos < total_bytes) {
                uint32_t t0 = buf[byte_pos] & 0x07;
                uint32_t t1 = buf[byte_pos] >> 5;
                if (t0 <= 2u * PARAM_ETA_S2) {
                    value0 = PARAM_Q + PARAM_ETA_S2 - (int32_t)t0;
                    have0 = 1;
                    count++;
                }
                if (t1 <= 2u * PARAM_ETA_S2) {
                    value1 = PARAM_Q + PARAM_ETA_S2 - (int32_t)t1;
                    have1 = 1;
                    count++;
                }
            }

            const int prefix = wp_kg_subwarp_exclusive_scan<SUBWARP_LANES>(count, mask, sublane);
            const int accepted = wp_kg_subwarp_sum<SUBWARP_LANES>(count, mask, leader_lane);

            if (have0) {
                const int out_idx = cur_ctr + prefix;
                if (out_idx < PARAM_N)
                    dst[out_idx] = value0;
            }
            if (have1) {
                const int out_idx = cur_ctr + prefix + have0;
                if (out_idx < PARAM_N)
                    dst[out_idx] = value1;
            }

            if (sublane == 0) {
                const int next_ctr = cur_ctr + accepted;
                *ctr_ptr = next_ctr < PARAM_N ? next_ctr : PARAM_N;
            }
            __syncwarp(mask);
        }

        cur_ctr = *ctr_ptr;
        if (cur_ctr >= PARAM_N)
            break;

        if (sublane == 0) {
            stream256_squeezeblocks(buf, 1, &state);
            *buflen_ptr = STREAM256_BLOCKBYTES;
        }
        __syncwarp(mask);
    }
#endif
}

template<int SUBWARP_LANES>
__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_matrix_a_coop_kernel(
    coeff_t * __restrict__ d_mat,
    const unsigned char * __restrict__ d_buf,
    int batch_count)
{
    __shared__ uint8_t sh_buf[WP_KG_MAX_SUBWARPS_PER_BLOCK][WP_KG_MATRIX_COOP_BUF_BYTES];
    __shared__ int sh_ctr[WP_KG_MAX_SUBWARPS_PER_BLOCK];
    __shared__ unsigned int sh_buflen[WP_KG_MAX_SUBWARPS_PER_BLOCK];

    const int lane_in_warp = threadIdx.x & (WP_KG_WARP_SIZE - 1);
    const int warp_local = threadIdx.x / WP_KG_WARP_SIZE;
    const int subwarp_base = lane_in_warp - (lane_in_warp & (SUBWARP_LANES - 1));
    const int sublane = lane_in_warp - subwarp_base;
    const int subwarps_per_warp = WP_KG_WARP_SIZE / SUBWARP_LANES;
    const int group_local = warp_local * subwarps_per_warp + (lane_in_warp / SUBWARP_LANES);
    const int polys_per_block = blockDim.x / SUBWARP_LANES;
    const int poly_global = blockIdx.x * polys_per_block + group_local;
    const int total_polys = batch_count * PARAM_K * PARAM_L;
    const unsigned long long mask = wp_kg_subwarp_mask<SUBWARP_LANES>(lane_in_warp);

    if (poly_global >= total_polys) return;

    const int inst = poly_global / (PARAM_K * PARAM_L);
    const int poly_local = poly_global % (PARAM_K * PARAM_L);
    const int row = poly_local / PARAM_L;
    const int col = poly_local % PARAM_L;

    const unsigned char *my_buf = d_buf + (size_t)inst * (2 * SEEDBYTES + CRHBYTES);
    const uint8_t *rho = my_buf;
    coeff_t *dst = d_mat + (size_t)inst * PARAM_K * PARAM_L * PARAM_N + (size_t)poly_local * PARAM_N;

    wp_kg_uniform_coop_sample_to<SUBWARP_LANES>(
        dst, rho, MATRIX_NONCE(row, col),
        sh_buf[group_local], &sh_ctr[group_local], &sh_buflen[group_local],
        mask, sublane, subwarp_base);
}

template<int SUBWARP_LANES>
__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_secret_eta_coop_kernel(
    coeff_t * __restrict__ d_s1,
    coeff_t * __restrict__ d_s1hat,
    coeff_t * __restrict__ d_s2,
    const unsigned char * __restrict__ d_buf,
    int batch_count)
{
    __shared__ uint8_t sh_buf[WP_KG_MAX_SUBWARPS_PER_BLOCK][WP_KG_ETA_COOP_BUF_BYTES];
    __shared__ int sh_ctr[WP_KG_MAX_SUBWARPS_PER_BLOCK];
    __shared__ unsigned int sh_buflen[WP_KG_MAX_SUBWARPS_PER_BLOCK];

    const int lane_in_warp = threadIdx.x & (WP_KG_WARP_SIZE - 1);
    const int warp_local = threadIdx.x / WP_KG_WARP_SIZE;
    const int subwarp_base = lane_in_warp - (lane_in_warp & (SUBWARP_LANES - 1));
    const int sublane = lane_in_warp - subwarp_base;
    const int subwarps_per_warp = WP_KG_WARP_SIZE / SUBWARP_LANES;
    const int group_local = warp_local * subwarps_per_warp + (lane_in_warp / SUBWARP_LANES);
    const int polys_per_block = blockDim.x / SUBWARP_LANES;
    const int poly_global = blockIdx.x * polys_per_block + group_local;
    const int total_polys = batch_count * (PARAM_L + PARAM_K);
    const unsigned long long mask = wp_kg_subwarp_mask<SUBWARP_LANES>(lane_in_warp);

    if (poly_global >= total_polys) return;

    const int inst = poly_global / (PARAM_L + PARAM_K);
    const int poly_local = poly_global % (PARAM_L + PARAM_K);
    const unsigned char *my_buf = d_buf + (size_t)inst * (2 * SEEDBYTES + CRHBYTES);

#if ALGORITHM == ALGO_MLDSA
    const uint8_t *eta_seed = my_buf + 2 * SEEDBYTES;
#elif ALGORITHM == ALGO_AIGIS
    const uint8_t *eta_seed = my_buf + 2 * SEEDBYTES;
#endif

    if (poly_local < PARAM_L) {
        const int j = poly_local;
        coeff_t *dst = d_s1 + (size_t)inst * PARAM_L * PARAM_N + (size_t)j * PARAM_N;
        coeff_t *dst_copy = d_s1hat ? (d_s1hat + (size_t)inst * PARAM_L * PARAM_N + (size_t)j * PARAM_N) : NULL;
#if ALGORITHM == ALGO_MLDSA
        wp_kg_eta_mldsa_coop_sample_to<SUBWARP_LANES>(
            dst, dst_copy, eta_seed, (uint16_t)j,
            PARAM_ETA_S1, POLY_UNIFORM_ETA1_NBLOCKS,
            sh_buf[group_local], &sh_ctr[group_local], &sh_buflen[group_local],
            mask, sublane, subwarp_base);
#elif ALGORITHM == ALGO_AIGIS
        wp_kg_eta1_aigis_coop_sample_to<SUBWARP_LANES>(
            dst, dst_copy, eta_seed, (uint16_t)j,
            sh_buf[group_local], &sh_ctr[group_local], &sh_buflen[group_local],
            mask, sublane, subwarp_base);
#endif
    } else {
        const int k = poly_local - PARAM_L;
        coeff_t *dst = d_s2 + (size_t)inst * PARAM_K * PARAM_N + (size_t)k * PARAM_N;
#if ALGORITHM == ALGO_MLDSA
        wp_kg_eta_mldsa_coop_sample_to<SUBWARP_LANES>(
            dst, NULL, eta_seed, (uint16_t)(PARAM_L + k),
            PARAM_ETA_S2, POLY_UNIFORM_ETA2_NBLOCKS,
            sh_buf[group_local], &sh_ctr[group_local], &sh_buflen[group_local],
            mask, sublane, subwarp_base);
#elif ALGORITHM == ALGO_AIGIS
        wp_kg_eta2_aigis_coop_sample_to<SUBWARP_LANES>(
            dst, eta_seed, (uint16_t)(PARAM_L + k),
            sh_buf[group_local], &sh_ctr[group_local], &sh_buflen[group_local],
            mask, sublane, subwarp_base);
#endif
    }
}

#if ALGORITHM == ALGO_AIGIS
template<int SUBWARP_LANES>
__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_secret_eta1_aigis_coop_kernel(
    coeff_t * __restrict__ d_s1,
    coeff_t * __restrict__ d_s1hat,
    const unsigned char * __restrict__ d_buf,
    int batch_count)
{
    __shared__ uint8_t sh_buf[WP_KG_MAX_SUBWARPS_PER_BLOCK][WP_KG_ETA_COOP_BUF_BYTES];
    __shared__ int sh_ctr[WP_KG_MAX_SUBWARPS_PER_BLOCK];
    __shared__ unsigned int sh_buflen[WP_KG_MAX_SUBWARPS_PER_BLOCK];

    const int lane_in_warp = threadIdx.x & (WP_KG_WARP_SIZE - 1);
    const int warp_local = threadIdx.x / WP_KG_WARP_SIZE;
    const int subwarp_base = lane_in_warp - (lane_in_warp & (SUBWARP_LANES - 1));
    const int sublane = lane_in_warp - subwarp_base;
    const int subwarps_per_warp = WP_KG_WARP_SIZE / SUBWARP_LANES;
    const int group_local = warp_local * subwarps_per_warp + (lane_in_warp / SUBWARP_LANES);
    const int polys_per_block = blockDim.x / SUBWARP_LANES;
    const int poly_global = blockIdx.x * polys_per_block + group_local;
    const int total_polys = batch_count * PARAM_L;
    const unsigned long long mask = wp_kg_subwarp_mask<SUBWARP_LANES>(lane_in_warp);

    if (poly_global >= total_polys) return;

    const int inst = poly_global / PARAM_L;
    const int j = poly_global % PARAM_L;
    const unsigned char *my_buf = d_buf + (size_t)inst * (2 * SEEDBYTES + CRHBYTES);
    const uint8_t *eta_seed = my_buf + 2 * SEEDBYTES;
    coeff_t *dst = d_s1 + (size_t)inst * PARAM_L * PARAM_N + (size_t)j * PARAM_N;
    coeff_t *dst_copy = d_s1hat ? (d_s1hat + (size_t)inst * PARAM_L * PARAM_N + (size_t)j * PARAM_N) : NULL;

    wp_kg_eta1_aigis_coop_sample_to<SUBWARP_LANES>(
        dst, dst_copy, eta_seed, (uint16_t)j,
        sh_buf[group_local], &sh_ctr[group_local], &sh_buflen[group_local],
        mask, sublane, subwarp_base);
}

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_secret_eta2_aigis_sample_kernel(
    coeff_t * __restrict__ d_s2,
    const unsigned char * __restrict__ d_buf,
    int batch_count)
{
    const int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) / WP_KG_WARP_SIZE;
    const int lane   = threadIdx.x & (WP_KG_WARP_SIZE - 1);

    if (warp_g >= batch_count) return;

    const unsigned char *my_buf = d_buf + (size_t)warp_g * (2 * SEEDBYTES + CRHBYTES);
    const uint8_t *eta_seed = my_buf + 2 * SEEDBYTES;

    for (int k = lane; k < PARAM_K; k += WP_KG_WARP_SIZE) {
        coeff_t *dst = d_s2 + (size_t)warp_g * PARAM_K * PARAM_N + (size_t)k * PARAM_N;
        poly_uniform_eta_s2_to(dst, eta_seed, (uint16_t)(PARAM_L + k));
    }
}
#endif

__global__ void batch_keygen_paper_rho_to_buf_kernel(
    unsigned char * __restrict__ d_buf,
    const unsigned char * __restrict__ d_shared_rho,
    int batch_count);

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_paper_secret_sample_split_kernel(
    coeff_t * __restrict__ d_s1,
    coeff_t * __restrict__ d_s1hat,
    coeff_t * __restrict__ d_s2,
    unsigned char * __restrict__ d_buf,
    const unsigned char * __restrict__ d_shared_rho,
    int batch_count);

static inline void launch_batch_keygen_matrix_a_active(
    coeff_t *d_mat,
    const unsigned char *d_buf,
    int batch_count,
    int nblk,
    hipStream_t stream)
{
#if BATCH_KEYGEN_MATRIX_A_COOP
#if BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP
#if BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP_LANES == 8
    {
        const int polys_per_block = WP_KG_TPB / 8;
        const int coop_nblk = (batch_count * PARAM_K * PARAM_L + polys_per_block - 1) / polys_per_block;
        batch_keygen_matrix_a_coop_kernel<8><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_mat, d_buf, batch_count);
    }
#elif BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP_LANES == 16
    {
        const int polys_per_block = WP_KG_TPB / 16;
        const int coop_nblk = (batch_count * PARAM_K * PARAM_L + polys_per_block - 1) / polys_per_block;
        batch_keygen_matrix_a_coop_kernel<16><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_mat, d_buf, batch_count);
    }
#else
#error Unsupported BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP_LANES
#endif
#else
    {
        const int polys_per_block = WP_KG_TPB / 32;
        const int coop_nblk = (batch_count * PARAM_K * PARAM_L + polys_per_block - 1) / polys_per_block;
        batch_keygen_matrix_a_coop_kernel<32><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_mat, d_buf, batch_count);
    }
#endif
#elif BATCH_KEYGEN_MATRIX_A_LANEOPT
    batch_keygen_matrix_a_laneopt_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        d_mat, d_buf, batch_count);
#else
    batch_keygen_matrix_a_sample_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        d_mat, d_buf, batch_count);
#endif
}

static inline void launch_batch_keygen_secret_eta_active_independent(
    coeff_t *d_s1,
    coeff_t *d_s2,
    const unsigned char *d_buf,
    int batch_count,
    int nblk,
    hipStream_t stream)
{
#if BATCH_KEYGEN_SECRET_ETA_COOP
#if ALGORITHM == ALGO_AIGIS && PARAM_ETA_S2 == 5 && BATCH_KEYGEN_SECRET_ETA_AIGIS5_SPLIT
#if BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 8
    {
        const int polys_per_block = WP_KG_TPB / 8;
        const int coop_nblk = (batch_count * PARAM_L + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta1_aigis_coop_kernel<8><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, NULL, d_buf, batch_count);
    }
#elif BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 16
    {
        const int polys_per_block = WP_KG_TPB / 16;
        const int coop_nblk = (batch_count * PARAM_L + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta1_aigis_coop_kernel<16><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, NULL, d_buf, batch_count);
    }
#elif BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 32
    {
        const int polys_per_block = WP_KG_TPB / 32;
        const int coop_nblk = (batch_count * PARAM_L + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta1_aigis_coop_kernel<32><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, NULL, d_buf, batch_count);
    }
#else
#error Unsupported BATCH_KEYGEN_SECRET_ETA_COOP_LANES
#endif
    batch_keygen_secret_eta2_aigis_sample_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        d_s2, d_buf, batch_count);
#else
#if BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 8
    {
        const int polys_per_block = WP_KG_TPB / 8;
        const int coop_nblk = (batch_count * (PARAM_L + PARAM_K) + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta_coop_kernel<8><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, NULL, d_s2, d_buf, batch_count);
    }
#elif BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 16
    {
        const int polys_per_block = WP_KG_TPB / 16;
        const int coop_nblk = (batch_count * (PARAM_L + PARAM_K) + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta_coop_kernel<16><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, NULL, d_s2, d_buf, batch_count);
    }
#elif BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 32
    {
        const int polys_per_block = WP_KG_TPB / 32;
        const int coop_nblk = (batch_count * (PARAM_L + PARAM_K) + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta_coop_kernel<32><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, NULL, d_s2, d_buf, batch_count);
    }
#else
#error Unsupported BATCH_KEYGEN_SECRET_ETA_COOP_LANES
#endif
#endif
#else
    batch_keygen_secret_sample_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        d_s1, d_s2, d_buf, batch_count);
#endif
}

static inline void launch_batch_keygen_paper_rho_active(
    unsigned char *d_buf,
    const unsigned char *d_shared_rho,
    int batch_count,
    hipStream_t stream)
{
    batch_keygen_paper_rho_to_buf_kernel<<<(batch_count * SEEDBYTES + BATCH_TPB - 1) / BATCH_TPB, BATCH_TPB, 0, stream>>>(
        d_buf, d_shared_rho, batch_count);
}

static inline void launch_batch_keygen_secret_eta_active_paper(
    coeff_t *d_s1,
    coeff_t *d_s1hat,
    coeff_t *d_s2,
    unsigned char *d_buf,
    const unsigned char *d_shared_rho,
    int batch_count,
    int nblk,
    hipStream_t stream)
{
#if BATCH_KEYGEN_SECRET_ETA_COOP
#if ALGORITHM == ALGO_AIGIS && PARAM_ETA_S2 == 5 && BATCH_KEYGEN_SECRET_ETA_AIGIS5_SPLIT
#if BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 8
    {
        const int polys_per_block = WP_KG_TPB / 8;
        const int coop_nblk = (batch_count * PARAM_L + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta1_aigis_coop_kernel<8><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, d_s1hat, d_buf, batch_count);
    }
#elif BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 16
    {
        const int polys_per_block = WP_KG_TPB / 16;
        const int coop_nblk = (batch_count * PARAM_L + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta1_aigis_coop_kernel<16><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, d_s1hat, d_buf, batch_count);
    }
#elif BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 32
    {
        const int polys_per_block = WP_KG_TPB / 32;
        const int coop_nblk = (batch_count * PARAM_L + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta1_aigis_coop_kernel<32><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, d_s1hat, d_buf, batch_count);
    }
#else
#error Unsupported BATCH_KEYGEN_SECRET_ETA_COOP_LANES
#endif
    batch_keygen_secret_eta2_aigis_sample_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        d_s2, d_buf, batch_count);
#else
#if BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 8
    {
        const int polys_per_block = WP_KG_TPB / 8;
        const int coop_nblk = (batch_count * (PARAM_L + PARAM_K) + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta_coop_kernel<8><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, d_s1hat, d_s2, d_buf, batch_count);
    }
#elif BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 16
    {
        const int polys_per_block = WP_KG_TPB / 16;
        const int coop_nblk = (batch_count * (PARAM_L + PARAM_K) + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta_coop_kernel<16><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, d_s1hat, d_s2, d_buf, batch_count);
    }
#elif BATCH_KEYGEN_SECRET_ETA_COOP_LANES == 32
    {
        const int polys_per_block = WP_KG_TPB / 32;
        const int coop_nblk = (batch_count * (PARAM_L + PARAM_K) + polys_per_block - 1) / polys_per_block;
        batch_keygen_secret_eta_coop_kernel<32><<<coop_nblk, WP_KG_TPB, 0, stream>>>(
            d_s1, d_s1hat, d_s2, d_buf, batch_count);
    }
#else
#error Unsupported BATCH_KEYGEN_SECRET_ETA_COOP_LANES
#endif
#endif
#else
    batch_keygen_paper_secret_sample_split_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        d_s1, d_s1hat, d_s2, d_buf, d_shared_rho, batch_count);
#endif
}

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_paper_shared_a_kernel(
    coeff_t * __restrict__ d_shared_mat,
    unsigned char * __restrict__ d_shared_rho,
    const unsigned char * __restrict__ d_base_seed)
{
    __shared__ unsigned char sh_rho[SEEDBYTES];

    int tid = threadIdx.x;
    if (tid == 0) {
#if ALGORITHM == ALGO_MLDSA
        uint8_t buf[2 * SEEDBYTES + CRHBYTES];
        for (int i = 0; i < SEEDBYTES; i++) buf[i] = d_base_seed[i];
        buf[SEEDBYTES]     = PARAM_K;
        buf[SEEDBYTES + 1] = PARAM_L;
        shake256(buf, 2 * SEEDBYTES + CRHBYTES, buf, SEEDBYTES + 2);
        for (int i = 0; i < SEEDBYTES; i++) sh_rho[i] = buf[i];
#elif ALGORITHM == ALGO_AIGIS
        uint8_t buf[3 * SEEDBYTES];
        shake256(buf, 3 * SEEDBYTES, d_base_seed, SEEDBYTES);
        for (int i = 0; i < SEEDBYTES; i++) sh_rho[i] = buf[SEEDBYTES + i];
#endif
        for (int i = 0; i < SEEDBYTES; i++) d_shared_rho[i] = sh_rho[i];
    }
    __syncthreads();

    const int total = PARAM_K * PARAM_L;
    for (int p = tid; p < total; p += blockDim.x) {
        poly tmp;
        int row = p / PARAM_L;
        int col = p % PARAM_L;
        poly_uniform(&tmp, sh_rho, MATRIX_NONCE(row, col));
        coeff_t *dst = d_shared_mat + (size_t)p * PARAM_N;
        for (int c = 0; c < PARAM_N; c++) dst[c] = tmp.coeffs[c];
    }
}

__global__ void batch_keygen_paper_rho_to_buf_kernel(
    unsigned char * __restrict__ d_buf,
    const unsigned char * __restrict__ d_shared_rho,
    int batch_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * SEEDBYTES;
    if (idx >= total) return;

    int inst = idx / SEEDBYTES;
    int off = idx % SEEDBYTES;
    d_buf[(size_t)inst * (2 * SEEDBYTES + CRHBYTES) + off] = d_shared_rho[off];
}

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_paper_secret_sample_split_kernel(
    coeff_t * __restrict__ d_s1,
    coeff_t * __restrict__ d_s1hat,
    coeff_t * __restrict__ d_s2,
    unsigned char * __restrict__ d_buf,
    const unsigned char * __restrict__ d_shared_rho,
    int batch_count)
{
    const int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) / WP_KG_WARP_SIZE;
    const int lane   = threadIdx.x & (WP_KG_WARP_SIZE - 1);

    if (warp_g >= batch_count) return;

    unsigned char *my_buf = d_buf + (size_t)warp_g * (2 * SEEDBYTES + CRHBYTES);
    if (lane == 0) {
        for (int i = 0; i < SEEDBYTES; i++) my_buf[i] = d_shared_rho[i];
    }
    __syncwarp();

    const int total = PARAM_L + PARAM_K;

#if ALGORITHM == ALGO_MLDSA
    const uint8_t *rhoprime = my_buf + 2 * SEEDBYTES;
#elif ALGORITHM == ALGO_AIGIS
    const uint8_t *eta_seed = my_buf + 2 * SEEDBYTES;
#endif

    for (int p = lane; p < total; p += WP_KG_WARP_SIZE) {
        if (p < PARAM_L) {
            int j = p;
            coeff_t *dst1 = d_s1 + (size_t)warp_g * PARAM_L * PARAM_N + (size_t)j * PARAM_N;
#if ALGORITHM == ALGO_MLDSA
            poly_uniform_eta_s1_to(dst1, rhoprime, j);
#elif ALGORITHM == ALGO_AIGIS
            poly_uniform_eta_s1_to(dst1, eta_seed, (uint16_t)j);
#endif
            coeff_t *dsth = d_s1hat + (size_t)warp_g * PARAM_L * PARAM_N + (size_t)j * PARAM_N;
            for (int c = 0; c < PARAM_N; c++) dsth[c] = dst1[c];
        } else {
            int k = p - PARAM_L;
            coeff_t *dst = d_s2 + (size_t)warp_g * PARAM_K * PARAM_N + (size_t)k * PARAM_N;
#if ALGORITHM == ALGO_MLDSA
            poly_uniform_eta_s2_to(dst, rhoprime, PARAM_L + k);
#elif ALGORITHM == ALGO_AIGIS
            poly_uniform_eta_s2_to(dst, eta_seed, (uint16_t)(PARAM_L + k));
#endif
        }
    }
}

__global__ void __launch_bounds__(WP_KG_TPB)
batch_keygen_paper_secret_sample_kernel(
    coeff_t * __restrict__ d_s1,
    coeff_t * __restrict__ d_s1hat,
    coeff_t * __restrict__ d_s2,
    unsigned char * __restrict__ d_buf,
    const unsigned char * __restrict__ d_base_seed,
    const unsigned char * __restrict__ d_shared_rho,
    int batch_count)
{
    __shared__ unsigned char sh_seeds[WP_KG_WARPS_BLOCK][WP_KG_SEED_BYTES];
    __shared__ unsigned char sh_rho[WP_KG_WARPS_BLOCK][SEEDBYTES];

    const int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) / WP_KG_WARP_SIZE;
    const int lane   = threadIdx.x & (WP_KG_WARP_SIZE - 1);
    const int warp_l = threadIdx.x / WP_KG_WARP_SIZE;

    if (warp_g >= batch_count) return;

    unsigned char *my_seeds = sh_seeds[warp_l];
    unsigned char *my_rho = sh_rho[warp_l];

    if (lane == 0) {
        uint8_t seed_in[SEEDBYTES];
        for (int i = 0; i < SEEDBYTES; i++) seed_in[i] = d_base_seed[i];
        seed_in[SEEDBYTES - 4] ^= (uint8_t)(warp_g);
        seed_in[SEEDBYTES - 3] ^= (uint8_t)(warp_g >> 8);
        seed_in[SEEDBYTES - 2] ^= (uint8_t)(warp_g >> 16);
        seed_in[SEEDBYTES - 1] ^= (uint8_t)(warp_g >> 24);

#if ALGORITHM == ALGO_MLDSA
        uint8_t buf[2 * SEEDBYTES + CRHBYTES];
        for (int i = 0; i < SEEDBYTES; i++) buf[i] = seed_in[i];
        buf[SEEDBYTES]     = PARAM_K;
        buf[SEEDBYTES + 1] = PARAM_L;
        shake256(buf, 2 * SEEDBYTES + CRHBYTES, buf, SEEDBYTES + 2);
        for (int i = 0; i < 2 * SEEDBYTES + CRHBYTES; i++) my_seeds[i] = buf[i];
#elif ALGORITHM == ALGO_AIGIS
        uint8_t buf[3 * SEEDBYTES];
        shake256(buf, 3 * SEEDBYTES, seed_in, SEEDBYTES);
        for (int i = 0; i < 3 * SEEDBYTES; i++) my_seeds[i] = buf[i];
#endif
        for (int i = 0; i < SEEDBYTES; i++) my_rho[i] = d_shared_rho[i];

        unsigned char *my_buf = d_buf + (size_t)warp_g * (2 * SEEDBYTES + CRHBYTES);
        for (int i = 0; i < SEEDBYTES; i++) my_buf[i] = my_rho[i];
#if ALGORITHM == ALGO_MLDSA
        const uint8_t *key = my_seeds + SEEDBYTES + CRHBYTES;
        const uint8_t *rhp = my_seeds + SEEDBYTES;
        for (int i = 0; i < SEEDBYTES; i++) my_buf[SEEDBYTES + i] = key[i];
        for (int i = 0; i < CRHBYTES; i++) my_buf[2 * SEEDBYTES + i] = rhp[i];
#elif ALGORITHM == ALGO_AIGIS
    const uint8_t *eta_seed = my_seeds;
        const uint8_t *key = my_seeds + 2 * SEEDBYTES;
        for (int i = 0; i < SEEDBYTES; i++) my_buf[SEEDBYTES + i] = key[i];
    for (int i = 0; i < SEEDBYTES; i++) my_buf[2 * SEEDBYTES + i] = eta_seed[i];
#endif
    }
    __syncwarp();

#if ALGORITHM == ALGO_MLDSA
    const uint8_t *rhoprime = my_seeds + SEEDBYTES;
#elif ALGORITHM == ALGO_AIGIS
    const uint8_t *eta_seed = my_seeds;
#endif

    const int total = PARAM_L + PARAM_K;
    for (int p = lane; p < total; p += WP_KG_WARP_SIZE) {
        if (p < PARAM_L) {
            int j = p;
            coeff_t *dst1 = d_s1 + (size_t)warp_g * PARAM_L * PARAM_N + (size_t)j * PARAM_N;
#if ALGORITHM == ALGO_MLDSA
            poly_uniform_eta_s1_to(dst1, rhoprime, j);
#elif ALGORITHM == ALGO_AIGIS
            poly_uniform_eta_s1_to(dst1, eta_seed, (uint16_t)j);
#endif
            coeff_t *dsth = d_s1hat + (size_t)warp_g * PARAM_L * PARAM_N + (size_t)j * PARAM_N;
            for (int c = 0; c < PARAM_N; c++) {
                dsth[c] = dst1[c];
            }
        } else {
            int k = p - PARAM_L;
            coeff_t *dst = d_s2 + (size_t)warp_g * PARAM_K * PARAM_N + (size_t)k * PARAM_N;
#if ALGORITHM == ALGO_MLDSA
            poly_uniform_eta_s2_to(dst, rhoprime, PARAM_L + k);
#elif ALGORITHM == ALGO_AIGIS
            poly_uniform_eta_s2_to(dst, eta_seed, (uint16_t)(PARAM_L + k));
#endif
        }
    }
}

/* ================================================================
 * 矩阵向量乘 kernel — 共用
 *
 * t[row] = Σ_{col} A[row][col] · s1hat[col]  (NTT 域)
 * grid: (batch_count, PARAM_K)
 * block: PARAM_N threads
 * ================================================================ */
__global__ void batch_keygen_matvec_kernel(
    coeff_t * __restrict__ d_t,
    const coeff_t * __restrict__ d_mat,
    const coeff_t * __restrict__ d_s1hat,
    int batch_count)
{
    int inst = blockIdx.x;
    int row  = blockIdx.y;
    if (inst >= batch_count) return;

    int tid = threadIdx.x;

    coeff2_t acc = 0;
    #pragma unroll
    for (int col = 0; col < PARAM_L; col++) {
        coeff_t a = d_mat[(size_t)inst * PARAM_K * PARAM_L * PARAM_N
                          + (row * PARAM_L + col) * PARAM_N + tid];
        coeff_t b = d_s1hat[(size_t)inst * PARAM_L * PARAM_N
                            + col * PARAM_N + tid];
        acc += (coeff2_t)a * b;
    }

    d_t[(size_t)inst * PARAM_K * PARAM_N + row * PARAM_N + tid] = (coeff_t)montgomery_reduce(acc);
}

__global__ void batch_keygen_matvec_shared_a_kernel(
    coeff_t * __restrict__ d_t,
    const coeff_t * __restrict__ d_shared_mat,
    const coeff_t * __restrict__ d_s1hat,
    int batch_count)
{
    int inst = blockIdx.x;
    int row  = blockIdx.y;
    if (inst >= batch_count) return;

    int tid = threadIdx.x;
    coeff2_t acc = 0;
    #pragma unroll
    for (int col = 0; col < PARAM_L; col++) {
        coeff_t a = d_shared_mat[(row * PARAM_L + col) * PARAM_N + tid];
        coeff_t b = d_s1hat[(size_t)inst * PARAM_L * PARAM_N
                            + col * PARAM_N + tid];
        acc += (coeff2_t)a * b;
    }
    d_t[(size_t)inst * PARAM_K * PARAM_N + row * PARAM_N + tid] = (coeff_t)montgomery_reduce(acc);
}

__global__ void batch_keygen_add_norm_kernel(
    coeff_t * __restrict__ d_t,
    const coeff_t * __restrict__ d_s2,
    int total_coeffs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) return;
    coeff_t v = d_t[idx] + d_s2[idx];
#if ALGORITHM == ALGO_MLDSA
    v = coeff_normalize(v);
#elif ALGORITHM == ALGO_AIGIS
    v = coeff_freeze_wide(v);
#endif
    d_t[idx] = v;
}

static inline void launch_batch_keygen_add_norm(coeff_t *d_t,
                                                const coeff_t *d_s2,
                                                int total_coeffs,
                                                hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_keygen_add_norm_kernel<<<nblk, BATCH_TPB, 0, stream>>>(d_t, d_s2, total_coeffs);
}

/* ================================================================
 * 打包 kernel — 参数位宽分叉
 * ================================================================ */
__global__ void __launch_bounds__(32)
batch_keygen_pack_kernel(
    unsigned char * __restrict__ d_pks,
    unsigned char * __restrict__ d_sks,
    coeff_t * __restrict__ d_t1_out,
    coeff_t * __restrict__ d_t0_out,
    unsigned char * __restrict__ d_tr_out,
    const coeff_t * __restrict__ d_t,
    const coeff_t * __restrict__ d_s1,
    const coeff_t * __restrict__ d_s2,
    const unsigned char * __restrict__ d_buf,
    int batch_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    polyveck t1_pk, t0_pk;
    polyvecl s1_pk;
    polyveck s2_pk;

    /* 从 flat 缓冲区加载 t 并在 pack 阶段就地 power2round */
    for (int i = 0; i < PARAM_K; i++) {
        const coeff_t *src = d_t + (size_t)idx * PARAM_K * PARAM_N + i * PARAM_N;
        for (int c = 0; c < PARAM_N; c++) {
            int32_t v = (int32_t)src[c];
#if ALGORITHM == ALGO_MLDSA
            v += (v >> 31) & PARAM_Q;
#endif
            t1_pk.vec[i].coeffs[c] = power2round(&t0_pk.vec[i].coeffs[c], v);
            if (d_t1_out)
                d_t1_out[(size_t)idx * PARAM_K * PARAM_N + (size_t)i * PARAM_N + c] = t1_pk.vec[i].coeffs[c];
            if (d_t0_out)
                d_t0_out[(size_t)idx * PARAM_K * PARAM_N + (size_t)i * PARAM_N + c] = t0_pk.vec[i].coeffs[c];
        }
    }

    /* 加载 s1, s2 */
    for (int i = 0; i < PARAM_L; i++) {
        const coeff_t *src = d_s1 + (size_t)idx * PARAM_L * PARAM_N + i * PARAM_N;
        for (int c = 0; c < PARAM_N; c++) s1_pk.vec[i].coeffs[c] = src[c];
    }
    for (int i = 0; i < PARAM_K; i++) {
        const coeff_t *src = d_s2 + (size_t)idx * PARAM_K * PARAM_N + i * PARAM_N;
        for (int c = 0; c < PARAM_N; c++) s2_pk.vec[i].coeffs[c] = src[c];
    }

    /* 从 d_buf 恢复 rho, key */
    const unsigned char *my_buf = d_buf + (size_t)idx * (2 * SEEDBYTES + CRHBYTES);
    uint8_t rho[SEEDBYTES], key_buf[SEEDBYTES];
    for (int i = 0; i < SEEDBYTES; i++) rho[i] = my_buf[i];
    for (int i = 0; i < SEEDBYTES; i++) key_buf[i] = my_buf[SEEDBYTES + i];

    /* pack pk */
    uint8_t *pk = d_pks + (size_t)idx * CRYPTO_PUBLICKEYBYTES;
    pack_pk(pk, rho, &t1_pk);

    /* hash pk → tr */
    uint8_t tr[TRBYTES];
    shake256(tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
    if (d_tr_out) {
        unsigned char *tr_dst = d_tr_out + (size_t)idx * TRBYTES;
        for (int i = 0; i < TRBYTES; i++) tr_dst[i] = tr[i];
    }

    /* pack sk */
    uint8_t *sk = d_sks + (size_t)idx * CRYPTO_SECRETKEYBYTES;
    pack_sk(sk, rho, key_buf, tr, &s1_pk, &s2_pk, &t0_pk);
}

__global__ void __launch_bounds__(32)
batch_keygen_pack_precomputed_kernel(
    unsigned char * __restrict__ d_pks,
    unsigned char * __restrict__ d_sks,
    const coeff_t * __restrict__ d_t1,
    const coeff_t * __restrict__ d_t0,
    const coeff_t * __restrict__ d_s1,
    const coeff_t * __restrict__ d_s2,
    const unsigned char * __restrict__ d_buf,
    unsigned char * __restrict__ d_tr_out,
    int batch_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    polyveck t1_pk, t0_pk;
    polyvecl s1_pk;
    polyveck s2_pk;

    for (int i = 0; i < PARAM_K; i++) {
        const coeff_t *src1 = d_t1 + (size_t)idx * PARAM_K * PARAM_N + i * PARAM_N;
        const coeff_t *src0 = d_t0 + (size_t)idx * PARAM_K * PARAM_N + i * PARAM_N;
        for (int c = 0; c < PARAM_N; c++) {
            t1_pk.vec[i].coeffs[c] = src1[c];
            t0_pk.vec[i].coeffs[c] = src0[c];
        }
    }

    for (int i = 0; i < PARAM_L; i++) {
        const coeff_t *src = d_s1 + (size_t)idx * PARAM_L * PARAM_N + i * PARAM_N;
        for (int c = 0; c < PARAM_N; c++) s1_pk.vec[i].coeffs[c] = src[c];
    }
    for (int i = 0; i < PARAM_K; i++) {
        const coeff_t *src = d_s2 + (size_t)idx * PARAM_K * PARAM_N + i * PARAM_N;
        for (int c = 0; c < PARAM_N; c++) s2_pk.vec[i].coeffs[c] = src[c];
    }

    const unsigned char *my_buf = d_buf + (size_t)idx * (2 * SEEDBYTES + CRHBYTES);
    uint8_t rho[SEEDBYTES], key_buf[SEEDBYTES];
    for (int i = 0; i < SEEDBYTES; i++) rho[i] = my_buf[i];
    for (int i = 0; i < SEEDBYTES; i++) key_buf[i] = my_buf[SEEDBYTES + i];

    uint8_t *pk = d_pks + (size_t)idx * CRYPTO_PUBLICKEYBYTES;
    pack_pk(pk, rho, &t1_pk);

    uint8_t tr[TRBYTES];
    shake256(tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
    if (d_tr_out) {
        unsigned char *tr_dst = d_tr_out + (size_t)idx * TRBYTES;
        for (int i = 0; i < TRBYTES; i++) tr_dst[i] = tr[i];
    }

    uint8_t *sk = d_sks + (size_t)idx * CRYPTO_SECRETKEYBYTES;
    pack_sk(sk, rho, key_buf, tr, &s1_pk, &s2_pk, &t0_pk);
}

#ifndef BATCH_KEYGEN_TR_HASH_FIXED
#define BATCH_KEYGEN_TR_HASH_FIXED 1
#endif

#if BATCH_KEYGEN_TR_HASH_FIXED
static __device__ __noinline__ void batch_keygen_shake256_tr_pk(
    uint8_t *out,
    const uint8_t *pk)
{
    keccak_state state;
    keccak_absorb_once(state.s, SHAKE256_RATE, pk, CRYPTO_PUBLICKEYBYTES, 0x1F);
    KeccakF1600_StatePermute(state.s);

    const int whole_words = TRBYTES / 8;
    for (int i = 0; i < whole_words; i++) {
        store64(out + 8 * i, state.s[i]);
    }

    const int tail_bytes = TRBYTES & 7;
    if (tail_bytes) {
        uint64_t tail_word = state.s[whole_words];
        for (int i = 0; i < tail_bytes; i++) {
            out[whole_words * 8 + i] = (uint8_t)(tail_word >> (8 * i));
        }
    }
}
#endif

static __device__ __noinline__ void batch_keygen_pack_header_task(
    unsigned char * __restrict__ d_pks,
    unsigned char * __restrict__ d_sks,
    const unsigned char * __restrict__ d_buf,
    int task_id)
{
    int inst = task_id / (2 * SEEDBYTES);
    int off  = task_id - inst * (2 * SEEDBYTES);
    const unsigned char *buf = d_buf + (size_t)inst * (2 * SEEDBYTES + CRHBYTES);
    unsigned char *pk = d_pks + (size_t)inst * CRYPTO_PUBLICKEYBYTES;
    unsigned char *sk = d_sks + (size_t)inst * CRYPTO_SECRETKEYBYTES;

    if (off < SEEDBYTES) {
        unsigned char rho = buf[off];
        pk[off] = rho;
        sk[off] = rho;
    } else {
        sk[off] = buf[off];
    }
}

static __device__ __noinline__ void batch_keygen_pack_t1_task(
    unsigned char * __restrict__ d_pks,
    const coeff_t * __restrict__ d_t1,
    int task_id)
{
#if POLYT1_PACKED_BITS == 10
    const int groups_per_poly = PARAM_N / 4;
#else
    const int groups_per_poly = PARAM_N;
#endif
    int inst = task_id / (PARAM_K * groups_per_poly);
    int rem = task_id - inst * (PARAM_K * groups_per_poly);
    int poly_idx = rem / groups_per_poly;
    int group = rem - poly_idx * groups_per_poly;
    unsigned char *pk = d_pks + (size_t)inst * CRYPTO_PUBLICKEYBYTES;

#if POLYT1_PACKED_BITS == 10
    const coeff_t *src = d_t1 + (size_t)inst * PARAM_K * PARAM_N
                       + (size_t)poly_idx * PARAM_N;
    unsigned char *dst = pk + SEEDBYTES + (size_t)poly_idx * POLYT1_PACKEDBYTES;
    uint32_t t0 = (uint32_t)src[4 * group + 0];
    uint32_t t1 = (uint32_t)src[4 * group + 1];
    uint32_t t2 = (uint32_t)src[4 * group + 2];
    uint32_t t3 = (uint32_t)src[4 * group + 3];
    dst[5 * group + 0] = (uint8_t)t0;
    dst[5 * group + 1] = (uint8_t)((t0 >> 8) | (t1 << 2));
    dst[5 * group + 2] = (uint8_t)((t1 >> 6) | (t2 << 4));
    dst[5 * group + 3] = (uint8_t)((t2 >> 4) | (t3 << 6));
    dst[5 * group + 4] = (uint8_t)(t3 >> 2);
#elif POLYT1_PACKED_BITS == 8
    const int coeff_idx = group;
    pk[SEEDBYTES + (size_t)poly_idx * POLYT1_PACKEDBYTES + coeff_idx] =
        (uint8_t)d_t1[(size_t)inst * PARAM_K * PARAM_N + rem];
#else
    #error Unsupported POLYT1_PACKED_BITS
#endif
}

static __device__ __noinline__ void batch_keygen_pack_s1_task(
    unsigned char * __restrict__ d_sks,
    const coeff_t * __restrict__ d_s1,
    int task_id)
{
#if SETA1BITS == 2
    const int groups_per_poly = PARAM_N / 4;
#elif SETA1BITS == 3
    const int groups_per_poly = PARAM_N / 8;
#else
    const int groups_per_poly = PARAM_N / 2;
#endif
    int inst = task_id / (PARAM_L * groups_per_poly);
    int rem = task_id - inst * (PARAM_L * groups_per_poly);
    int poly_idx = rem / groups_per_poly;
    int group = rem - poly_idx * groups_per_poly;

    const coeff_t *src = d_s1 + (size_t)inst * PARAM_L * PARAM_N
                       + (size_t)poly_idx * PARAM_N;
    unsigned char *dst = d_sks + (size_t)inst * CRYPTO_SECRETKEYBYTES
                       + (2 * SEEDBYTES + TRBYTES)
                       + (size_t)poly_idx * POLYETA1_PACKEDBYTES;

#if SETA1BITS == 2
    uint8_t t0 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[4 * group + 0]);
    uint8_t t1 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[4 * group + 1]);
    uint8_t t2 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[4 * group + 2]);
    uint8_t t3 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[4 * group + 3]);
    dst[group] = t0 | (t1 << 2) | (t2 << 4) | (t3 << 6);
#elif SETA1BITS == 3
    uint8_t t0 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[8 * group + 0]);
    uint8_t t1 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[8 * group + 1]);
    uint8_t t2 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[8 * group + 2]);
    uint8_t t3 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[8 * group + 3]);
    uint8_t t4 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[8 * group + 4]);
    uint8_t t5 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[8 * group + 5]);
    uint8_t t6 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[8 * group + 6]);
    uint8_t t7 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[8 * group + 7]);
    dst[3 * group + 0] = t0 | (t1 << 3) | (t2 << 6);
    dst[3 * group + 1] = (t2 >> 2) | (t3 << 1) | (t4 << 4) | (t5 << 7);
    dst[3 * group + 2] = (t5 >> 1) | (t6 << 2) | (t7 << 5);
#elif SETA1BITS == 4
    uint8_t t0 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[2 * group + 0]);
    uint8_t t1 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S1 - src[2 * group + 1]);
    dst[group] = t0 | (t1 << 4);
#endif
}

static __device__ __noinline__ void batch_keygen_pack_s2_task(
    unsigned char * __restrict__ d_sks,
    const coeff_t * __restrict__ d_s2,
    int task_id)
{
#if SETA2BITS == 3
    const int groups_per_poly = PARAM_N / 8;
#else
    const int groups_per_poly = PARAM_N / 2;
#endif
    int inst = task_id / (PARAM_K * groups_per_poly);
    int rem = task_id - inst * (PARAM_K * groups_per_poly);
    int poly_idx = rem / groups_per_poly;
    int group = rem - poly_idx * groups_per_poly;

    const coeff_t *src = d_s2 + (size_t)inst * PARAM_K * PARAM_N
                       + (size_t)poly_idx * PARAM_N;
    unsigned char *dst = d_sks + (size_t)inst * CRYPTO_SECRETKEYBYTES
                       + (2 * SEEDBYTES + TRBYTES)
                       + (size_t)PARAM_L * POLYETA1_PACKEDBYTES
                       + (size_t)poly_idx * POLYETA2_PACKEDBYTES;

#if SETA2BITS == 3
    uint8_t t0 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[8 * group + 0]);
    uint8_t t1 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[8 * group + 1]);
    uint8_t t2 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[8 * group + 2]);
    uint8_t t3 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[8 * group + 3]);
    uint8_t t4 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[8 * group + 4]);
    uint8_t t5 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[8 * group + 5]);
    uint8_t t6 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[8 * group + 6]);
    uint8_t t7 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[8 * group + 7]);
    dst[3 * group + 0] = t0 | (t1 << 3) | (t2 << 6);
    dst[3 * group + 1] = (t2 >> 2) | (t3 << 1) | (t4 << 4) | (t5 << 7);
    dst[3 * group + 2] = (t5 >> 1) | (t6 << 2) | (t7 << 5);
#elif SETA2BITS == 4
    uint8_t t0 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[2 * group + 0]);
    uint8_t t1 = (uint8_t)(COEFF_BIAS + PARAM_ETA_S2 - src[2 * group + 1]);
    dst[group] = t0 | (t1 << 4);
#endif
}

static __device__ __noinline__ void batch_keygen_pack_t0_task(
    unsigned char * __restrict__ d_sks,
    const coeff_t * __restrict__ d_t0,
    int task_id)
{
#if PARAM_D == 13
    const int groups_per_poly = PARAM_N / 8;
#else
    const int groups_per_poly = PARAM_N / 4;
#endif
    int inst = task_id / (PARAM_K * groups_per_poly);
    int rem = task_id - inst * (PARAM_K * groups_per_poly);
    int poly_idx = rem / groups_per_poly;
    int group = rem - poly_idx * groups_per_poly;

    const coeff_t *src = d_t0 + (size_t)inst * PARAM_K * PARAM_N
                       + (size_t)poly_idx * PARAM_N;
    unsigned char *dst = d_sks + (size_t)inst * CRYPTO_SECRETKEYBYTES
                       + (2 * SEEDBYTES + TRBYTES)
                       + (size_t)PARAM_L * POLYETA1_PACKEDBYTES
                       + (size_t)PARAM_K * POLYETA2_PACKEDBYTES
                       + (size_t)poly_idx * POLYT0_PACKEDBYTES;

#if PARAM_D == 13
    uint32_t t[8];
    for (int j = 0; j < 8; j++)
        t[j] = COEFF_BIAS + (1 << (PARAM_D - 1)) - src[8 * group + j];
    dst[13 * group +  0]  =  t[0];
    dst[13 * group +  1]  =  t[0] >> 8;
    dst[13 * group +  1] |=  t[1] << 5;
    dst[13 * group +  2]  =  t[1] >> 3;
    dst[13 * group +  3]  =  t[1] >> 11;
    dst[13 * group +  3] |=  t[2] << 2;
    dst[13 * group +  4]  =  t[2] >> 6;
    dst[13 * group +  4] |=  t[3] << 7;
    dst[13 * group +  5]  =  t[3] >> 1;
    dst[13 * group +  6]  =  t[3] >> 9;
    dst[13 * group +  6] |=  t[4] << 4;
    dst[13 * group +  7]  =  t[4] >> 4;
    dst[13 * group +  8]  =  t[4] >> 12;
    dst[13 * group +  8] |=  t[5] << 1;
    dst[13 * group +  9]  =  t[5] >> 7;
    dst[13 * group +  9] |=  t[6] << 6;
    dst[13 * group + 10]  =  t[6] >> 2;
    dst[13 * group + 11]  =  t[6] >> 10;
    dst[13 * group + 11] |=  t[7] << 3;
    dst[13 * group + 12]  =  t[7] >> 5;
#elif PARAM_D == 14
    uint32_t t[4];
    for (int j = 0; j < 4; j++)
        t[j] = COEFF_BIAS + (1 << (PARAM_D - 1)) - src[4 * group + j];
    dst[7 * group + 0]  =  t[0];
    dst[7 * group + 1]  =  t[0] >> 8;
    dst[7 * group + 1] |=  t[1] << 6;
    dst[7 * group + 2]  =  t[1] >> 2;
    dst[7 * group + 3]  =  t[1] >> 10;
    dst[7 * group + 3] |=  t[2] << 4;
    dst[7 * group + 4]  =  t[2] >> 4;
    dst[7 * group + 5]  =  t[2] >> 12;
    dst[7 * group + 5] |=  t[3] << 2;
    dst[7 * group + 6]  =  t[3] >> 6;
#endif
}

__global__ void batch_keygen_pack_header_kernel(
    unsigned char * __restrict__ d_pks,
    unsigned char * __restrict__ d_sks,
    const unsigned char * __restrict__ d_buf,
    int total_bytes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bytes) return;

    batch_keygen_pack_header_task(d_pks, d_sks, d_buf, idx);
}

__global__ void batch_keygen_pack_t1_kernel(
    unsigned char * __restrict__ d_pks,
    const coeff_t * __restrict__ d_t1,
    int total_groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_groups) return;

    batch_keygen_pack_t1_task(d_pks, d_t1, idx);
}

__global__ void batch_keygen_pack_s1_kernel(
    unsigned char * __restrict__ d_sks,
    const coeff_t * __restrict__ d_s1,
    int total_groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_groups) return;

    batch_keygen_pack_s1_task(d_sks, d_s1, idx);
}

__global__ void batch_keygen_pack_s2_kernel(
    unsigned char * __restrict__ d_sks,
    const coeff_t * __restrict__ d_s2,
    int total_groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_groups) return;

    batch_keygen_pack_s2_task(d_sks, d_s2, idx);
}

__global__ void batch_keygen_pack_t0_kernel(
    unsigned char * __restrict__ d_sks,
    const coeff_t * __restrict__ d_t0,
    int total_groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_groups) return;

    batch_keygen_pack_t0_task(d_sks, d_t0, idx);
}

__global__ void batch_keygen_pack_body_kernel(
    unsigned char * __restrict__ d_pks,
    unsigned char * __restrict__ d_sks,
    const coeff_t * __restrict__ d_t1,
    const coeff_t * __restrict__ d_t0,
    const coeff_t * __restrict__ d_s1,
    const coeff_t * __restrict__ d_s2,
    const unsigned char * __restrict__ d_buf,
    int total_tasks,
    int header_end,
    int t1_end,
    int s1_end,
    int s2_end)
{
    int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_id >= total_tasks) return;

    if (task_id < header_end) {
        batch_keygen_pack_header_task(d_pks, d_sks, d_buf, task_id);
    } else if (task_id < t1_end) {
        batch_keygen_pack_t1_task(d_pks, d_t1, task_id - header_end);
    } else if (task_id < s1_end) {
        batch_keygen_pack_s1_task(d_sks, d_s1, task_id - t1_end);
    } else if (task_id < s2_end) {
        batch_keygen_pack_s2_task(d_sks, d_s2, task_id - s1_end);
    } else {
        batch_keygen_pack_t0_task(d_sks, d_t0, task_id - s2_end);
    }
}

__global__ void batch_keygen_tr_hash_kernel(
    unsigned char * __restrict__ d_pks,
    unsigned char * __restrict__ d_sks,
    unsigned char * __restrict__ d_tr_out,
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;

    const size_t pk_base = (size_t)inst * CRYPTO_PUBLICKEYBYTES;
    const size_t sk_base = (size_t)inst * CRYPTO_SECRETKEYBYTES;
    const size_t tr_base = (size_t)inst * TRBYTES;
    const unsigned char *pk = d_pks + pk_base;
    unsigned char *sk_tr = d_sks + sk_base + 2 * SEEDBYTES;
    unsigned char *tr_out = d_tr_out ? (d_tr_out + tr_base) : NULL;
    uint8_t tr[TRBYTES];
#if BATCH_KEYGEN_TR_HASH_FIXED
    batch_keygen_shake256_tr_pk(tr, pk);
#else
    shake256(tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
#endif
    for (int i = 0; i < TRBYTES; i++) {
        sk_tr[i] = tr[i];
        if (tr_out)
            tr_out[i] = tr[i];
    }
}

__global__ void __launch_bounds__(32)
batch_keygen_pack_fused_tr_kernel(
    unsigned char * __restrict__ d_pks,
    unsigned char * __restrict__ d_sks,
    const coeff_t * __restrict__ d_t1,
    const coeff_t * __restrict__ d_t0,
    const coeff_t * __restrict__ d_s1,
    const coeff_t * __restrict__ d_s2,
    const unsigned char * __restrict__ d_buf,
    unsigned char * __restrict__ d_tr_out,
    int batch_count,
    int header_tasks_per_inst,
    int t1_tasks_per_inst,
    int s1_tasks_per_inst,
    int s2_tasks_per_inst,
    int t0_tasks_per_inst)
{
    const int inst = blockIdx.x;
    if (inst >= batch_count) return;

    const int tid = threadIdx.x;
    const int header_end = header_tasks_per_inst;
    const int t1_end = header_end + t1_tasks_per_inst;
    const int s1_end = t1_end + s1_tasks_per_inst;
    const int s2_end = s1_end + s2_tasks_per_inst;
    const int total_tasks = s2_end + t0_tasks_per_inst;

    for (int task = tid; task < total_tasks; task += blockDim.x) {
        if (task < header_end) {
            batch_keygen_pack_header_task(
                d_pks, d_sks, d_buf,
                inst * header_tasks_per_inst + task);
        } else if (task < t1_end) {
            batch_keygen_pack_t1_task(
                d_pks, d_t1,
                inst * t1_tasks_per_inst + (task - header_end));
        } else if (task < s1_end) {
            batch_keygen_pack_s1_task(
                d_sks, d_s1,
                inst * s1_tasks_per_inst + (task - t1_end));
        } else if (task < s2_end) {
            batch_keygen_pack_s2_task(
                d_sks, d_s2,
                inst * s2_tasks_per_inst + (task - s1_end));
        } else {
            batch_keygen_pack_t0_task(
                d_sks, d_t0,
                inst * t0_tasks_per_inst + (task - s2_end));
        }
    }

    __threadfence_block();
    __syncthreads();

    if (tid == 0) {
        const size_t pk_base = (size_t)inst * CRYPTO_PUBLICKEYBYTES;
        const size_t sk_base = (size_t)inst * CRYPTO_SECRETKEYBYTES;
        const size_t tr_base = (size_t)inst * TRBYTES;
        unsigned char *pk = d_pks + pk_base;
        unsigned char *sk_tr = d_sks + sk_base + 2 * SEEDBYTES;
        unsigned char *tr_out = d_tr_out ? (d_tr_out + tr_base) : NULL;
        polyveck t1_pk;
        uint8_t rho[SEEDBYTES];

        const unsigned char *my_buf = d_buf + (size_t)inst * (2 * SEEDBYTES + CRHBYTES);
        for (int i = 0; i < SEEDBYTES; i++) rho[i] = my_buf[i];

        for (int i = 0; i < PARAM_K; i++) {
            const coeff_t *src1 = d_t1 + (size_t)inst * PARAM_K * PARAM_N + i * PARAM_N;
            for (int c = 0; c < PARAM_N; c++) {
                t1_pk.vec[i].coeffs[c] = src1[c];
            }
        }

        pack_pk(pk, rho, &t1_pk);

        uint8_t tr[TRBYTES];
        shake256(tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
        for (int i = 0; i < TRBYTES; i++) {
            sk_tr[i] = tr[i];
            if (tr_out)
                tr_out[i] = tr[i];
        }
    }
}

#ifndef BATCH_KEYGEN_PACK_USE_REFERENCE
#define BATCH_KEYGEN_PACK_USE_REFERENCE 0
#endif

#ifndef BATCH_KEYGEN_PACK_PROFILE_SPLIT
#define BATCH_KEYGEN_PACK_PROFILE_SPLIT 0
#endif

#ifndef BATCH_KEYGEN_TR_HASH_EXPERIMENTAL
#define BATCH_KEYGEN_TR_HASH_EXPERIMENTAL 0
#endif

#ifndef BATCH_KEYGEN_PACK_FUSED_TR
#define BATCH_KEYGEN_PACK_FUSED_TR 0
#endif

static inline void launch_batch_keygen_tr_hash(
    unsigned char *d_pks,
    unsigned char *d_sks,
    unsigned char *d_tr,
    int batch_count,
    hipStream_t stream = 0)
{
#if BATCH_KEYGEN_TR_HASH_FIXED
    const int tpb = 32;
#else
    const int tpb = 128;
#endif
    const int nblk = (batch_count + tpb - 1) / tpb;
#if BATCH_KEYGEN_TR_HASH_EXPERIMENTAL
    batch_keygen_tr_hash_kernel<<<nblk, tpb, 0, stream>>>(
        d_pks, d_sks, d_tr, batch_count);
#else
    batch_keygen_tr_hash_kernel<<<nblk, tpb, 0, stream>>>(
        d_pks, d_sks, d_tr, batch_count);
#endif
}

static inline void launch_batch_keygen_pack_reference(
    unsigned char *d_pks,
    unsigned char *d_sks,
    const coeff_t *d_t1,
    const coeff_t *d_t0,
    const coeff_t *d_s1,
    const coeff_t *d_s2,
    const unsigned char *d_buf,
    unsigned char *d_tr,
    int batch_count,
    hipStream_t stream = 0)
{
    int tpb = 32;
    int nblk = (batch_count + tpb - 1) / tpb;
    batch_keygen_pack_precomputed_kernel<<<nblk, tpb, 0, stream>>>(
        d_pks, d_sks, d_t1, d_t0, d_s1, d_s2, d_buf, d_tr, batch_count);
}

static inline void launch_batch_keygen_pack_standard(
    unsigned char *d_pks,
    unsigned char *d_sks,
    const coeff_t *d_t1,
    const coeff_t *d_t0,
    const coeff_t *d_s1,
    const coeff_t *d_s2,
    const unsigned char *d_buf,
    unsigned char *d_tr,
    int batch_count,
    hipStream_t stream = 0,
    KeygenProfile *profile = NULL)
{
#if BATCH_KEYGEN_PACK_USE_REFERENCE
    launch_batch_keygen_pack_reference(
        d_pks, d_sks, d_t1, d_t0, d_s1, d_s2, d_buf, d_tr, batch_count, stream);
#else
    hipEvent_t ev0 = NULL, ev1 = NULL, ev_inner0 = NULL, ev_inner1 = NULL;
    if (profile) {
        hipEventCreate(&ev0);
        hipEventCreate(&ev1);
        hipEventCreate(&ev_inner0);
        hipEventCreate(&ev_inner1);
        hipEventRecord(ev_inner0, stream);
    }

    const int tpb = BATCH_TPB;
    const int header_tasks_per_inst = 2 * SEEDBYTES;
    int total_header = batch_count * header_tasks_per_inst;

#if POLYT1_PACKED_BITS == 10
    const int t1_groups_per_poly = PARAM_N / 4;
#else
    const int t1_groups_per_poly = PARAM_N;
#endif
    const int t1_tasks_per_inst = PARAM_K * t1_groups_per_poly;
    int total_t1_groups = batch_count * t1_tasks_per_inst;

#if SETA1BITS == 2
    const int s1_groups_per_poly = PARAM_N / 4;
#elif SETA1BITS == 3
    const int s1_groups_per_poly = PARAM_N / 8;
#else
    const int s1_groups_per_poly = PARAM_N / 2;
#endif
    const int s1_tasks_per_inst = PARAM_L * s1_groups_per_poly;
    int total_s1_groups = batch_count * s1_tasks_per_inst;

#if SETA2BITS == 3
    const int s2_groups_per_poly = PARAM_N / 8;
#else
    const int s2_groups_per_poly = PARAM_N / 2;
#endif
    const int s2_tasks_per_inst = PARAM_K * s2_groups_per_poly;
    int total_s2_groups = batch_count * s2_tasks_per_inst;

#if PARAM_D == 13
    const int t0_groups_per_poly = PARAM_N / 8;
#else
    const int t0_groups_per_poly = PARAM_N / 4;
#endif
    const int t0_tasks_per_inst = PARAM_K * t0_groups_per_poly;
    int total_t0_groups = batch_count * t0_tasks_per_inst;

#if BATCH_KEYGEN_PACK_FUSED_TR
    if (profile) hipEventRecord(ev0, stream);
    batch_keygen_pack_fused_tr_kernel<<<batch_count, 32, 0, stream>>>(
        d_pks, d_sks, d_t1, d_t0, d_s1, d_s2, d_buf, d_tr, batch_count,
        header_tasks_per_inst, t1_tasks_per_inst, s1_tasks_per_inst,
        s2_tasks_per_inst, t0_tasks_per_inst);
    if (profile) {
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->pack_fused_ms, ev0, ev1);
        hipEventRecord(ev_inner1, stream);
        hipEventSynchronize(ev_inner1);
        keygen_profile_add(&profile->pack_inner_ms, ev_inner0, ev_inner1);
        hipEventDestroy(ev0);
        hipEventDestroy(ev1);
        hipEventDestroy(ev_inner0);
        hipEventDestroy(ev_inner1);
    }
#else

    if (profile && BATCH_KEYGEN_PACK_PROFILE_SPLIT) {
        if (profile) hipEventRecord(ev0, stream);
        batch_keygen_pack_header_kernel<<<(total_header + tpb - 1) / tpb, tpb, 0, stream>>>(
            d_pks, d_sks, d_buf, total_header);
        if (profile) {
            hipEventRecord(ev1, stream);
            hipEventSynchronize(ev1);
            keygen_profile_add(&profile->pack_header_ms, ev0, ev1);
        }

        if (profile) hipEventRecord(ev0, stream);
        batch_keygen_pack_t1_kernel<<<(total_t1_groups + tpb - 1) / tpb, tpb, 0, stream>>>(
            d_pks, d_t1, total_t1_groups);
        if (profile) {
            hipEventRecord(ev1, stream);
            hipEventSynchronize(ev1);
            keygen_profile_add(&profile->pack_t1_ms, ev0, ev1);
        }

        if (profile) hipEventRecord(ev0, stream);
        batch_keygen_pack_s1_kernel<<<(total_s1_groups + tpb - 1) / tpb, tpb, 0, stream>>>(
            d_sks, d_s1, total_s1_groups);
        batch_keygen_pack_s2_kernel<<<(total_s2_groups + tpb - 1) / tpb, tpb, 0, stream>>>(
            d_sks, d_s2, total_s2_groups);
        if (profile) {
            hipEventRecord(ev1, stream);
            hipEventSynchronize(ev1);
            keygen_profile_add(&profile->pack_eta_ms, ev0, ev1);
        }

        if (profile) hipEventRecord(ev0, stream);
        batch_keygen_pack_t0_kernel<<<(total_t0_groups + tpb - 1) / tpb, tpb, 0, stream>>>(
            d_sks, d_t0, total_t0_groups);
        if (profile) {
            hipEventRecord(ev1, stream);
            hipEventSynchronize(ev1);
            keygen_profile_add(&profile->pack_t0_ms, ev0, ev1);
            hipEventRecord(ev1, stream);
            hipEventSynchronize(ev1);
            keygen_profile_add(&profile->pack_body_ms, ev_inner0, ev1);
        }
    } else {
        int header_end = total_header;
        int t1_end = header_end + total_t1_groups;
        int s1_end = t1_end + total_s1_groups;
        int s2_end = s1_end + total_s2_groups;
        int total_body_tasks = s2_end + total_t0_groups;

        if (profile) hipEventRecord(ev0, stream);
        batch_keygen_pack_body_kernel<<<(total_body_tasks + tpb - 1) / tpb, tpb, 0, stream>>>(
            d_pks, d_sks, d_t1, d_t0, d_s1, d_s2, d_buf,
            total_body_tasks, header_end, t1_end, s1_end, s2_end);
        if (profile) {
            hipEventRecord(ev1, stream);
            hipEventSynchronize(ev1);
            keygen_profile_add(&profile->pack_body_ms, ev0, ev1);
        }
    }

    if (profile) hipEventRecord(ev0, stream);
    launch_batch_keygen_tr_hash(d_pks, d_sks, d_tr, batch_count, stream);
    if (profile) {
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->tr_hash_ms, ev0, ev1);
        hipEventRecord(ev_inner1, stream);
        hipEventSynchronize(ev_inner1);
        keygen_profile_add(&profile->pack_inner_ms, ev_inner0, ev_inner1);
        hipEventDestroy(ev0);
        hipEventDestroy(ev1);
        hipEventDestroy(ev_inner0);
        hipEventDestroy(ev_inner1);
    }
#endif
#endif
}

__global__ void batch_keygen_shiftl_copy_kernel(
    coeff_t * __restrict__ d_dst,
    const coeff_t * __restrict__ d_src,
    int total_coeffs,
    int shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) return;
    d_dst[idx] = d_src[idx] << shift;
}

static inline void launch_batch_keygen_shiftl_copy(coeff_t *d_dst,
                                                   const coeff_t *d_src,
                                                   int total_coeffs,
                                                   int shift,
                                                   hipStream_t stream = 0) {
    int nblk = (total_coeffs + BATCH_TPB - 1) / BATCH_TPB;
    batch_keygen_shiftl_copy_kernel<<<nblk, BATCH_TPB, 0, stream>>>(
        d_dst, d_src, total_coeffs, shift);
}

static inline void batch_keygen_finalize_material(BatchKeygenBuffers *buf,
                                                  int batch_count,
                                                  hipStream_t stream = 0) {
    const int total_k = batch_count * PARAM_K * PARAM_N;
    hipMemcpyAsync(buf->d_s2_ntt, buf->d_s2,
                    (size_t)total_k * sizeof(coeff_t),
                    hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(buf->d_t0_ntt, buf->d_t0,
                    (size_t)total_k * sizeof(coeff_t),
                    hipMemcpyDeviceToDevice, stream);
    launch_batch_keygen_shiftl_copy(buf->d_t1_hat, buf->d_t1,
                                    total_k, PARAM_D, stream);
    launch_batch_ntt(buf->d_s2_ntt, batch_count * PARAM_K, stream);
    launch_batch_ntt(buf->d_t0_ntt, batch_count * PARAM_K, stream);
    launch_batch_ntt(buf->d_t1_hat, batch_count * PARAM_K, stream);
}

__global__ void batch_keygen_material_to_precomp_kernel(
    precomp_t * __restrict__ pc,
    const coeff_t * __restrict__ d_mat,
    const coeff_t * __restrict__ d_s1_ntt,
    const coeff_t * __restrict__ d_s2_ntt,
    const coeff_t * __restrict__ d_t0_ntt,
    const unsigned char * __restrict__ d_buf,
    const unsigned char * __restrict__ d_tr,
    int inst,
    int mat_shared)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const size_t mat_base = mat_shared ? 0 :
        (size_t)inst * PARAM_K * PARAM_L * PARAM_N;
    for (int k = 0; k < PARAM_K; k++) {
        for (int l = 0; l < PARAM_L; l++) {
            const coeff_t *src = d_mat + mat_base
                + (size_t)(k * PARAM_L + l) * PARAM_N;
            for (int c = 0; c < PARAM_N; c++)
                pc->mat[k].vec[l].coeffs[c] = src[c];
        }
    }

    const size_t s1_base = (size_t)inst * PARAM_L * PARAM_N;
    for (int l = 0; l < PARAM_L; l++) {
        const coeff_t *src = d_s1_ntt + s1_base + (size_t)l * PARAM_N;
        for (int c = 0; c < PARAM_N; c++)
            pc->s1_ntt.vec[l].coeffs[c] = src[c];
    }

    const size_t k_base = (size_t)inst * PARAM_K * PARAM_N;
    for (int k = 0; k < PARAM_K; k++) {
        const coeff_t *s2 = d_s2_ntt + k_base + (size_t)k * PARAM_N;
        const coeff_t *t0 = d_t0_ntt + k_base + (size_t)k * PARAM_N;
        for (int c = 0; c < PARAM_N; c++) {
            pc->s2_ntt.vec[k].coeffs[c] = s2[c];
            pc->t0_ntt.vec[k].coeffs[c] = t0[c];
        }
    }

    const unsigned char *my_buf = d_buf + (size_t)inst * (2 * SEEDBYTES + CRHBYTES);
    for (int i = 0; i < SEEDBYTES; i++) pc->key[i] = my_buf[SEEDBYTES + i];
    const unsigned char *tr = d_tr + (size_t)inst * TRBYTES;
    for (int i = 0; i < TRBYTES; i++) pc->tr[i] = tr[i];
}

__global__ void batch_keygen_material_to_verify_kernel(
    coeff_t * __restrict__ d_vmat,
    coeff_t * __restrict__ d_vt1_hat,
    unsigned char * __restrict__ d_vtr,
    const coeff_t * __restrict__ d_mat,
    const coeff_t * __restrict__ d_t1_hat,
    const unsigned char * __restrict__ d_tr,
    int inst,
    int mat_shared)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int mat_total = PARAM_K * PARAM_L * PARAM_N;
    if (idx < mat_total) {
        size_t src_idx = mat_shared ? (size_t)idx :
            (size_t)inst * PARAM_K * PARAM_L * PARAM_N + idx;
        d_vmat[idx] = d_mat[src_idx];
    }
    const int t1_total = PARAM_K * PARAM_N;
    if (idx < t1_total) {
        d_vt1_hat[idx] = d_t1_hat[(size_t)inst * PARAM_K * PARAM_N + idx];
    }
    if (idx < TRBYTES) {
        d_vtr[idx] = d_tr[(size_t)inst * TRBYTES + idx];
    }
}

/* ================================================================
 * Host API — 缓冲区分配/释放
 * ================================================================ */

static int batch_keygen_alloc(BatchKeygenBuffers *buf, int max_batch) {
    memset(buf, 0, sizeof(*buf));
    buf->max_batch = max_batch;
    size_t B = max_batch;
    size_t N = PARAM_N;

#define BKG_TRY(ptr, sz) do { \
    if (hipMalloc(&(ptr), (sz)) != hipSuccess) { hipGetLastError(); return -1; } \
} while(0)

    BKG_TRY(buf->d_mat,    B * PARAM_K * PARAM_L * N * sizeof(coeff_t));
    BKG_TRY(buf->d_s1,     B * PARAM_L * N * sizeof(coeff_t));
    BKG_TRY(buf->d_s1hat,  B * PARAM_L * N * sizeof(coeff_t));
    BKG_TRY(buf->d_s2,     B * PARAM_K * N * sizeof(coeff_t));
    BKG_TRY(buf->d_t,      B * PARAM_K * N * sizeof(coeff_t));
    BKG_TRY(buf->d_t1,     B * PARAM_K * N * sizeof(coeff_t));
    BKG_TRY(buf->d_t0,     B * PARAM_K * N * sizeof(coeff_t));
    BKG_TRY(buf->d_t1_hat, B * PARAM_K * N * sizeof(coeff_t));
    BKG_TRY(buf->d_s2_ntt, B * PARAM_K * N * sizeof(coeff_t));
    BKG_TRY(buf->d_t0_ntt, B * PARAM_K * N * sizeof(coeff_t));
    BKG_TRY(buf->d_tr,     B * TRBYTES);
    BKG_TRY(buf->d_pks,    B * CRYPTO_PUBLICKEYBYTES);
    BKG_TRY(buf->d_sks,    B * CRYPTO_SECRETKEYBYTES);
    BKG_TRY(buf->d_buf,    B * (2 * SEEDBYTES + CRHBYTES));

#undef BKG_TRY
    return 0;
}

static void batch_keygen_free(BatchKeygenBuffers *buf) {
    hipFree(buf->d_mat);  hipFree(buf->d_s1);
    hipFree(buf->d_s1hat); hipFree(buf->d_s2);
    hipFree(buf->d_t);    hipFree(buf->d_t1);
    hipFree(buf->d_t0);   hipFree(buf->d_t1_hat);
    hipFree(buf->d_s2_ntt); hipFree(buf->d_t0_ntt);
    hipFree(buf->d_tr);   hipFree(buf->d_pks);
    hipFree(buf->d_sks);  hipFree(buf->d_buf);
    memset(buf, 0, sizeof(*buf));
}

/* ================================================================
 * 批量密钥生成 pipeline (1 warp/instance 算子级并行采样)
 *
 * 采样阶段: 1 warp (32 线程) per instance 并行生成所有多项式
 *   - Aigis-sig3: 30+5+6=41 个多项式, 约 2 轮
 *   - ML-DSA-87:  56+7+8=71 个多项式, 约 3 轮
 * ================================================================ */
static int batch_keygen_pipeline_warp(
    unsigned char *d_pks,
    unsigned char *d_sks,
    const unsigned char *d_seeds,
    BatchKeygenBuffers *buf,
    int batch_count,
    hipStream_t stream = 0,
    int produce_material = 1)
{

    if (batch_count <= 0 || batch_count > buf->max_batch) return -1;
    const int N = PARAM_N;

    /* [1] 算子级并行采样: 1 warp per instance, 32× 并行 SHAKE 调用 */
    {
        int nwarps  = batch_count;                           /* 每 instance 1 warp */
        int nthreads = nwarps * WP_KG_WARP_SIZE;            /* 总线程数 */
        int nblk    = (nthreads + WP_KG_TPB - 1) / WP_KG_TPB;
        batch_keygen_warp_sample_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
            buf->d_mat, buf->d_s1, buf->d_s2,
            buf->d_buf, d_seeds, batch_count);
    }

    /* [2] copy s1 → s1hat */
    /* [3] NTT(s1hat) — shared-memory batch */
    hipMemcpyAsync(buf->d_s1hat, buf->d_s1,
                    (size_t)batch_count * PARAM_L * N * sizeof(coeff_t),
                    hipMemcpyDeviceToDevice,
                    stream);
    launch_batch_ntt(buf->d_s1hat, batch_count * PARAM_L, stream);

    /* [4] 矩阵向量乘 */
    {
        dim3 grid(batch_count, PARAM_K);
        batch_keygen_matvec_kernel<<<grid, N, 0, stream>>>(
            buf->d_t, buf->d_mat, buf->d_s1hat, batch_count);
    }

    /* [5] reduce + INVNTT */
    launch_batch_reduce(buf->d_t, batch_count * PARAM_K * N, stream);
    launch_batch_invntt(buf->d_t, batch_count * PARAM_K, stream);

    /* [6] t += s2 */
    launch_batch_add(buf->d_t, buf->d_t, buf->d_s2,
                     batch_count * PARAM_K * N, stream);

    /* [6.5] normalize */
#if ALGORITHM == ALGO_MLDSA
    launch_batch_caddq(buf->d_t, batch_count * PARAM_K * N, stream);
#elif ALGORITHM == ALGO_AIGIS
    launch_batch_freeze_wide(buf->d_t, batch_count * PARAM_K * N, stream);
#endif

    /* [7] 打包 pk, sk */
    launch_batch_power2round(buf->d_t1, buf->d_t0, buf->d_t,
                             batch_count * PARAM_K * N, stream);
    launch_batch_keygen_pack_standard(d_pks, d_sks,
                                      buf->d_t1, buf->d_t0,
                                      buf->d_s1, buf->d_s2,
                                      buf->d_buf, buf->d_tr,
                                      batch_count, stream);
    if (produce_material)
        batch_keygen_finalize_material(buf, batch_count, stream);

    return 0;
}

static inline void launch_batch_keygen_sample_independent(
    BatchKeygenBuffers *buf,
    const unsigned char *d_seeds,
    int batch_count,
    KeygenProfile *profile,
    hipEvent_t ev0,
    hipEvent_t ev1,
    hipStream_t stream)
{
    int nwarps = batch_count;
    int nthreads = nwarps * WP_KG_WARP_SIZE;
    int nblk = (nthreads + WP_KG_TPB - 1) / WP_KG_TPB;

#if BATCH_KEYGEN_MATRIX_A_COOP || BATCH_KEYGEN_MATRIX_A_LANEOPT || \
    BATCH_KEYGEN_SECRET_ETA_COOP || BATCH_KEYGEN_MATRIX_A_FAST || \
    BATCH_KEYGEN_SECRET_ETA_FAST || \
    BATCH_KEYGEN_SAMPLE_SPLIT_FAST
    if (profile) {
        hipEvent_t sample_ev0 = NULL, sample_ev1 = NULL;
        hipEventCreate(&sample_ev0);
        hipEventCreate(&sample_ev1);
        hipEventRecord(sample_ev0, stream);

        hipEventRecord(ev0, stream);
        batch_keygen_seed_expand_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
            buf->d_buf, d_seeds, batch_count);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->seed_expand_ms, ev0, ev1);

        hipEventRecord(ev0, stream);
        launch_batch_keygen_matrix_a_active(
            buf->d_mat, buf->d_buf, batch_count, nblk, stream);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->matrix_a_sample_ms, ev0, ev1);

        hipEventRecord(ev0, stream);
        launch_batch_keygen_secret_eta_active_independent(
            buf->d_s1, buf->d_s2, buf->d_buf, batch_count, nblk, stream);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->secret_eta_sample_ms, ev0, ev1);

        hipEventRecord(sample_ev1, stream);
        hipEventSynchronize(sample_ev1);
        keygen_profile_add(&profile->sample_ms, sample_ev0, sample_ev1);
        keygen_profile_finalize_sample(
            profile,
            profile->seed_expand_ms + profile->matrix_a_sample_ms + profile->secret_eta_sample_ms);
#if BATCH_KEYGEN_MATRIX_A_COOP
        profile->matrix_a_coop_ms = profile->matrix_a_sample_ms;
#if BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP
        profile->matrix_a_coop_lanes = BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP_LANES;
#else
        profile->matrix_a_coop_lanes = 32;
#endif
#endif
#if BATCH_KEYGEN_SECRET_ETA_COOP
        profile->secret_eta_coop_ms = profile->secret_eta_sample_ms;
        profile->secret_eta_coop_lanes = BATCH_KEYGEN_SECRET_ETA_COOP_LANES;
#endif

        hipEventDestroy(sample_ev0);
        hipEventDestroy(sample_ev1);
    } else {
        batch_keygen_seed_expand_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
            buf->d_buf, d_seeds, batch_count);
        launch_batch_keygen_matrix_a_active(
            buf->d_mat, buf->d_buf, batch_count, nblk, stream);
        launch_batch_keygen_secret_eta_active_independent(
            buf->d_s1, buf->d_s2, buf->d_buf, batch_count, nblk, stream);
    }
#else
    if (profile) hipEventRecord(ev0, stream);
    batch_keygen_warp_sample_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        buf->d_mat, buf->d_s1, buf->d_s2,
        buf->d_buf, d_seeds, batch_count);
    if (profile) {
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->sample_ms, ev0, ev1);
        keygen_profile_finalize_sample(profile, profile->sample_ms);
    }
#endif
}

static inline void launch_batch_keygen_sample_paper(
    BatchKeygenBuffers *buf,
    const unsigned char *d_seeds,
    const unsigned char *d_shared_rho,
    int batch_count,
    KeygenProfile *profile,
    hipEvent_t ev0,
    hipEvent_t ev1,
    hipStream_t stream)
{
    int nwarps = batch_count;
    int nthreads = nwarps * WP_KG_WARP_SIZE;
    int nblk = (nthreads + WP_KG_TPB - 1) / WP_KG_TPB;

#if BATCH_KEYGEN_SECRET_ETA_COOP || BATCH_KEYGEN_SECRET_ETA_FAST || \
    BATCH_KEYGEN_SAMPLE_SPLIT_FAST
    if (profile) {
        hipEvent_t sample_ev0 = NULL, sample_ev1 = NULL;
        hipEventCreate(&sample_ev0);
        hipEventCreate(&sample_ev1);
        hipEventRecord(sample_ev0, stream);

        hipEventRecord(ev0, stream);
        batch_keygen_seed_expand_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
            buf->d_buf, d_seeds, batch_count);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->seed_expand_ms, ev0, ev1);

        hipEventRecord(ev0, stream);
        launch_batch_keygen_paper_rho_active(
            buf->d_buf, d_shared_rho, batch_count, stream);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->matrix_a_sample_ms, ev0, ev1);

        hipEventRecord(ev0, stream);
        launch_batch_keygen_secret_eta_active_paper(
            buf->d_s1, buf->d_s1hat, buf->d_s2,
            buf->d_buf, d_shared_rho, batch_count, nblk, stream);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->secret_eta_sample_ms, ev0, ev1);

        hipEventRecord(sample_ev1, stream);
        hipEventSynchronize(sample_ev1);
        keygen_profile_add(&profile->sample_ms, sample_ev0, sample_ev1);
        keygen_profile_finalize_sample(
            profile,
            profile->seed_expand_ms + profile->matrix_a_sample_ms + profile->secret_eta_sample_ms);
#if BATCH_KEYGEN_SECRET_ETA_COOP
        profile->secret_eta_coop_ms = profile->secret_eta_sample_ms;
        profile->secret_eta_coop_lanes = BATCH_KEYGEN_SECRET_ETA_COOP_LANES;
#endif

        hipEventDestroy(sample_ev0);
        hipEventDestroy(sample_ev1);
    } else {
        batch_keygen_seed_expand_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
            buf->d_buf, d_seeds, batch_count);
        launch_batch_keygen_paper_rho_active(
            buf->d_buf, d_shared_rho, batch_count, stream);
        launch_batch_keygen_secret_eta_active_paper(
            buf->d_s1, buf->d_s1hat, buf->d_s2,
            buf->d_buf, d_shared_rho, batch_count, nblk, stream);
    }
#else
    if (profile) hipEventRecord(ev0, stream);
    batch_keygen_paper_secret_sample_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        buf->d_s1, buf->d_s1hat, buf->d_s2,
        buf->d_buf, d_seeds, d_shared_rho, batch_count);
    if (profile) {
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->sample_ms, ev0, ev1);
        keygen_profile_finalize_sample(profile, profile->sample_ms);
    }
#endif
}

static int batch_keygen_sample_only_independent(
    BatchKeygenBuffers *buf,
    const unsigned char *d_seeds,
    int batch_count,
    KeygenSampleOnlyProfile *profile,
    hipStream_t stream = 0)
{
    if (!profile || batch_count <= 0 || batch_count > buf->max_batch) return -1;

    hipEvent_t ev0 = NULL, ev1 = NULL;
    KeygenProfile active_profile;
    int nwarps = batch_count;
    int nthreads = nwarps * WP_KG_WARP_SIZE;
    int nblk = (nthreads + WP_KG_TPB - 1) / WP_KG_TPB;

    keygen_sample_only_profile_clear(profile);
    keygen_profile_clear(&active_profile);
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    hipEventRecord(ev0, stream);
    batch_keygen_warp_sample_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        buf->d_mat, buf->d_s1, buf->d_s2,
        buf->d_buf, d_seeds, batch_count);
    hipEventRecord(ev1, stream);
    hipEventSynchronize(ev1);
    keygen_profile_add(&profile->old_fused_ms, ev0, ev1);

    launch_batch_keygen_sample_independent(
        buf, d_seeds, batch_count, &active_profile, ev0, ev1, stream);
    profile->split_seed_ms = active_profile.seed_expand_ms;
    profile->split_matrix_a_ms = active_profile.matrix_a_sample_ms;
    profile->split_eta_ms = active_profile.secret_eta_sample_ms;
    profile->split_total_ms = active_profile.sample_ms;
    profile->split_launch_gap_ms = active_profile.sample_launch_gap_ms;
    profile->split_matrix_a_coop_ms = active_profile.matrix_a_coop_ms;
    profile->split_eta_coop_ms = active_profile.secret_eta_coop_ms;
    profile->split_matrix_a_coop_lanes = active_profile.matrix_a_coop_lanes;
    profile->split_eta_coop_lanes = active_profile.secret_eta_coop_lanes;

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    return 0;
}

static int batch_keygen_sample_only_paper(
    BatchKeygenBuffers *buf,
    const unsigned char *d_seeds,
    unsigned char *d_shared_rho,
    int batch_count,
    KeygenSampleOnlyProfile *profile,
    hipStream_t stream = 0)
{
    if (!profile || batch_count <= 0 || batch_count > buf->max_batch) return -1;

    hipEvent_t ev0 = NULL, ev1 = NULL;
    KeygenProfile active_profile;
    int nwarps = batch_count;
    int nthreads = nwarps * WP_KG_WARP_SIZE;
    int nblk = (nthreads + WP_KG_TPB - 1) / WP_KG_TPB;

    keygen_sample_only_profile_clear(profile);
    keygen_profile_clear(&active_profile);
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    hipEventRecord(ev0, stream);
    batch_keygen_paper_shared_a_kernel<<<1, WP_KG_TPB, 0, stream>>>(
        buf->d_mat, d_shared_rho, d_seeds);
    hipEventRecord(ev1, stream);
    hipEventSynchronize(ev1);
    keygen_profile_add(&profile->shared_a_ms, ev0, ev1);

    hipEventRecord(ev0, stream);
    batch_keygen_paper_secret_sample_kernel<<<nblk, WP_KG_TPB, 0, stream>>>(
        buf->d_s1, buf->d_s1hat, buf->d_s2,
        buf->d_buf, d_seeds, d_shared_rho, batch_count);
    hipEventRecord(ev1, stream);
    hipEventSynchronize(ev1);
    keygen_profile_add(&profile->old_fused_ms, ev0, ev1);

    launch_batch_keygen_sample_paper(
        buf, d_seeds, d_shared_rho, batch_count,
        &active_profile, ev0, ev1, stream);
    profile->split_seed_ms = active_profile.seed_expand_ms;
    profile->split_matrix_a_ms = active_profile.matrix_a_sample_ms;
    profile->split_eta_ms = active_profile.secret_eta_sample_ms;
    profile->split_total_ms = active_profile.sample_ms;
    profile->split_launch_gap_ms = active_profile.sample_launch_gap_ms;
    profile->split_matrix_a_coop_ms = active_profile.matrix_a_coop_ms;
    profile->split_eta_coop_ms = active_profile.secret_eta_coop_ms;
    profile->split_matrix_a_coop_lanes = active_profile.matrix_a_coop_lanes;
    profile->split_eta_coop_lanes = active_profile.secret_eta_coop_lanes;

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    return 0;
}

static int batch_keygen_pipeline_warp_opt(
    unsigned char *d_pks,
    unsigned char *d_sks,
    const unsigned char *d_seeds,
    BatchKeygenBuffers *buf,
    int batch_count,
    KeygenProfile *profile = NULL,
    hipStream_t stream = 0,
    int produce_material = 1)
{

    if (batch_count <= 0 || batch_count > buf->max_batch) return -1;
    const int N = PARAM_N;
    hipEvent_t ev0 = NULL, ev1 = NULL;
    if (profile) {
        keygen_profile_clear(profile);
        hipEventCreate(&ev0);
        hipEventCreate(&ev1);
    }

    launch_batch_keygen_sample_independent(
        buf, d_seeds, batch_count, profile, ev0, ev1, stream);

    if (profile) hipEventRecord(ev0, stream);
    hipMemcpyAsync(buf->d_s1hat, buf->d_s1,
                    (size_t)batch_count * PARAM_L * N * sizeof(coeff_t),
                    hipMemcpyDeviceToDevice,
                    stream);
    if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->copy_ms, ev0, ev1); }

    if (profile) hipEventRecord(ev0, stream);
    launch_batch_ntt(buf->d_s1hat, batch_count * PARAM_L, stream);
    if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->ntt_ms, ev0, ev1); }

    {
        dim3 grid(batch_count, PARAM_K);
        if (profile) hipEventRecord(ev0, stream);
        batch_keygen_matvec_kernel<<<grid, N, 0, stream>>>(
            buf->d_t, buf->d_mat, buf->d_s1hat, batch_count);
        if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->matvec_ms, ev0, ev1); }
    }

    if (profile) hipEventRecord(ev0, stream);
    launch_batch_reduce(buf->d_t, batch_count * PARAM_K * N, stream);
    launch_batch_invntt(buf->d_t, batch_count * PARAM_K, stream);
    launch_batch_keygen_add_norm(buf->d_t, buf->d_s2,
                                 batch_count * PARAM_K * N, stream);
    if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->post_ms, ev0, ev1); }

    if (profile) hipEventRecord(ev0, stream);
    launch_batch_power2round(buf->d_t1, buf->d_t0, buf->d_t,
                             batch_count * PARAM_K * N, stream);
    if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->p2r_ms, ev0, ev1); }

    if (profile) hipEventRecord(ev0, stream);
    launch_batch_keygen_pack_standard(d_pks, d_sks,
                                      buf->d_t1, buf->d_t0,
                                      buf->d_s1, buf->d_s2,
                                      buf->d_buf, buf->d_tr,
                                      batch_count, stream, profile);
    if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->pack_ms, ev0, ev1); }

    if (produce_material) {
        if (profile) hipEventRecord(ev0, stream);
        batch_keygen_finalize_material(buf, batch_count, stream);
        if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->material_ms, ev0, ev1); }
    }

    if (profile) {
        hipEventDestroy(ev0);
        hipEventDestroy(ev1);
    }
    return 0;
}

static int batch_keygen_create_shared_rho_a(
    BatchKeygenBuffers *buf,
    unsigned char *d_shared_rho,
    const unsigned char *d_base_seed,
    KeygenProfile *profile = NULL)
{
    hipEvent_t ev0 = NULL, ev1 = NULL;
    if (profile) {
        hipEventCreate(&ev0);
        hipEventCreate(&ev1);
        hipEventRecord(ev0);
    }
    batch_keygen_paper_shared_a_kernel<<<1, WP_KG_TPB>>>(
        buf->d_mat, d_shared_rho, d_base_seed);
    if (profile) {
        hipEventRecord(ev1);
        hipEventSynchronize(ev1);
        keygen_profile_add(&profile->shared_a_ms, ev0, ev1);
        hipEventDestroy(ev0);
        hipEventDestroy(ev1);
    }
    return 0;
}

static int batch_keygen_pipeline_paper_shared_rho_a(
    unsigned char *d_pks,
    unsigned char *d_sks,
    const unsigned char *d_seeds,
    const unsigned char *d_shared_rho,
    BatchKeygenBuffers *buf,
    int batch_count,
    KeygenProfile *profile = NULL,
    hipStream_t stream = 0,
    int produce_material = 1)
{

    if (batch_count <= 0 || batch_count > buf->max_batch) return -1;
    const int N = PARAM_N;
    hipEvent_t ev0 = NULL, ev1 = NULL;
    float shared_keep = profile ? profile->shared_a_ms : 0.0f;
    if (profile) {
        keygen_profile_clear(profile);
        profile->shared_a_ms = shared_keep;
        hipEventCreate(&ev0);
        hipEventCreate(&ev1);
    }

    launch_batch_keygen_sample_paper(
        buf, d_seeds, d_shared_rho, batch_count, profile, ev0, ev1, stream);

    if (profile) hipEventRecord(ev0, stream);
    launch_batch_ntt(buf->d_s1hat, batch_count * PARAM_L, stream);
    if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->ntt_ms, ev0, ev1); }

    {
        dim3 grid(batch_count, PARAM_K);
        if (profile) hipEventRecord(ev0, stream);
        batch_keygen_matvec_shared_a_kernel<<<grid, N, 0, stream>>>(
            buf->d_t, buf->d_mat, buf->d_s1hat, batch_count);
        if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->matvec_ms, ev0, ev1); }
    }

    if (profile) hipEventRecord(ev0, stream);
    launch_batch_reduce(buf->d_t, batch_count * PARAM_K * N, stream);
    launch_batch_invntt(buf->d_t, batch_count * PARAM_K, stream);
    launch_batch_keygen_add_norm(buf->d_t, buf->d_s2,
                                 batch_count * PARAM_K * N, stream);
    if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->post_ms, ev0, ev1); }

    if (profile) hipEventRecord(ev0, stream);
    launch_batch_power2round(buf->d_t1, buf->d_t0, buf->d_t,
                             batch_count * PARAM_K * N, stream);
    if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->p2r_ms, ev0, ev1); }

    if (profile) hipEventRecord(ev0, stream);
    launch_batch_keygen_pack_standard(d_pks, d_sks,
                                      buf->d_t1, buf->d_t0,
                                      buf->d_s1, buf->d_s2,
                                      buf->d_buf, buf->d_tr,
                                      batch_count, stream, profile);
    if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->pack_ms, ev0, ev1); }

    if (produce_material) {
        if (profile) hipEventRecord(ev0, stream);
        batch_keygen_finalize_material(buf, batch_count, stream);
        if (profile) { hipEventRecord(ev1, stream); hipEventSynchronize(ev1); keygen_profile_add(&profile->material_ms, ev0, ev1); }
    }

    if (profile) {
        hipEventDestroy(ev0);
        hipEventDestroy(ev1);
    }
    return 0;
}

static int batch_keygen_compare_device_buffer(
    const void *d_ref,
    const void *d_cand,
    size_t total_bytes,
    size_t inst_stride_bytes,
    size_t elem_size,
    KeygenCompareStage stage,
    KeygenCompareResult *out)
{
    if (!d_ref || !d_cand || !out) return -1;
    if (total_bytes == 0) return 0;

    unsigned char *h_ref = (unsigned char *)malloc(total_bytes);
    unsigned char *h_cand = (unsigned char *)malloc(total_bytes);
    if (!h_ref || !h_cand) {
        free(h_ref);
        free(h_cand);
        return -1;
    }

    hipError_t err = hipMemcpy(h_ref, d_ref, total_bytes, hipMemcpyDeviceToHost);
    if (err == hipSuccess)
        err = hipMemcpy(h_cand, d_cand, total_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        free(h_ref);
        free(h_cand);
        return -1;
    }

    for (size_t byte_idx = 0; byte_idx < total_bytes; ++byte_idx) {
        if (h_ref[byte_idx] == h_cand[byte_idx]) continue;

        keygen_compare_result_clear(out);
        out->stage = stage;
        out->instance = inst_stride_bytes ? (int)(byte_idx / inst_stride_bytes) : 0;
        out->byte_offset = inst_stride_bytes ? (byte_idx % inst_stride_bytes) : byte_idx;
        out->element_offset = elem_size ? (out->byte_offset / elem_size) : out->byte_offset;

        if (elem_size == sizeof(coeff_t)) {
            size_t coeff_idx = byte_idx / sizeof(coeff_t);
            const coeff_t *ref_coeffs = (const coeff_t *)h_ref;
            const coeff_t *cand_coeffs = (const coeff_t *)h_cand;
            out->ref_value = ref_coeffs[coeff_idx];
            out->cand_value = cand_coeffs[coeff_idx];
        } else {
            out->ref_value = (int64_t)h_ref[byte_idx];
            out->cand_value = (int64_t)h_cand[byte_idx];
        }

        free(h_ref);
        free(h_cand);
        return 1;
    }

    free(h_ref);
    free(h_cand);
    return 0;
}

static int batch_keygen_compare_active_path(
    const unsigned char *d_seeds,
    int batch_count,
    int use_paper_shared_a,
    int sample_only,
    KeygenCompareResult *out,
    hipStream_t stream = 0)
{
    (void)d_seeds;
    (void)batch_count;
    (void)use_paper_shared_a;
    (void)sample_only;
    (void)out;
    (void)stream;
    return -1;
}


#endif /* BATCH_KEYGEN_CUH */
