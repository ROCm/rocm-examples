/*
 * batch_kem.cuh — GPU 批量 KEM 流水线
 *
 * 参考 mldsa和aigis-sig/batch_keygen.cuh 的优化架构:
 *   - Warp 协同采样: 1 warp = 1 实例 (并行矩阵展开 + 噪声采样)
 *   - 共享内存批量 NTT (batch_ntt_kernel, 1 block/poly)
 *   - 2D grid 矩阵向量乘 (batch_polyvec_matvec_kernel)
 *   - SoA 内存布局: data[poly_idx * batch_count * N + inst * N + coeff]
 *
 * 性能要点 (RTX 3050 Ti):
 *   - 最优 batch size: Keygen/Encaps=16K, Decaps=8K-16K
 *   - VRAM 限制: K^2 * B * N * sizeof(int16_t) ≤ 可用显存
 */

#ifndef BATCH_KEM_CUH
#define BATCH_KEM_CUH

#include "rocm_compat.h"
#include <stdint.h>
#include <string.h>
#include "params.h"
#include "reduce.cuh"
#include "fips202.cuh"
#include "ntt.cuh"
#include "poly.cuh"
#include "polyvec.cuh"
#include "cbd.cuh"
#include "kem.cuh"
#include "batch_ntt.cuh"
#include "batch_ops.cuh"

/* ================================================================
 *  缓冲区结构体
 * ================================================================ */
struct BatchKemBuffers {
    /* 批量 keygen/encaps 工作缓冲区 — SoA 布局 [poly_idx][inst][coeff] */
    int16_t *d_mat;      /* K*K * B * N — 矩阵 A (NTT 域) */
    int16_t *d_skpv;     /* K   * B * N — 私钥 s (NTT 域) */
    int16_t *d_pkpv;     /* K   * B * N — 公钥多项式 b */
    int16_t *d_e;        /* K   * B * N — keygen 误差 e */

    /* KEM 字节缓冲区 */
    uint8_t *d_pk_bytes; /* B * PARAM_PUBLICKEYBYTES */
    uint8_t *d_sk_bytes; /* B * PARAM_SECRETKEYBYTES */
    uint8_t *d_ct_bytes; /* B * PARAM_CIPHERTEXTBYTES */
    uint8_t *d_ss_bytes; /* B * PARAM_SSBYTES */

    /* 随机种子 */
    uint8_t *d_coins_kg; /* B * 2*SYMBYTES — keygen 种子 */
    uint8_t *d_coins_enc;/* B * SYMBYTES   — encaps 种子 */

    uint8_t *d_publicseed_kg;
    uint8_t *d_noiseseed_kg;

    int max_batch;
};

static inline void batch_kem_alloc(BatchKemBuffers *buf, int max_batch)
{
    buf->max_batch = max_batch;
    cudaMalloc(&buf->d_mat,       (size_t)PARAM_K * PARAM_K * max_batch * PARAM_N * sizeof(int16_t));
    cudaMalloc(&buf->d_skpv,      (size_t)PARAM_K * max_batch * PARAM_N * sizeof(int16_t));
    cudaMalloc(&buf->d_pkpv,      (size_t)PARAM_K * max_batch * PARAM_N * sizeof(int16_t));
    cudaMalloc(&buf->d_e,         (size_t)PARAM_K * max_batch * PARAM_N * sizeof(int16_t));
    cudaMalloc(&buf->d_pk_bytes,  (size_t)max_batch * PARAM_PUBLICKEYBYTES);
    cudaMalloc(&buf->d_sk_bytes,  (size_t)max_batch * PARAM_SECRETKEYBYTES);
    cudaMalloc(&buf->d_ct_bytes,  (size_t)max_batch * PARAM_CIPHERTEXTBYTES);
    cudaMalloc(&buf->d_ss_bytes,  (size_t)max_batch * PARAM_SSBYTES);
    cudaMalloc(&buf->d_coins_kg,  (size_t)max_batch * 2 * PARAM_SYMBYTES);
    cudaMalloc(&buf->d_coins_enc, (size_t)max_batch * PARAM_SYMBYTES);
    cudaMalloc(&buf->d_publicseed_kg, (size_t)max_batch * PARAM_SYMBYTES);
    cudaMalloc(&buf->d_noiseseed_kg, (size_t)max_batch * PARAM_SYMBYTES);
}

static inline void batch_kem_free(BatchKemBuffers *buf)
{
    cudaFree(buf->d_mat);
    cudaFree(buf->d_skpv);
    cudaFree(buf->d_pkpv);
    cudaFree(buf->d_e);
    cudaFree(buf->d_pk_bytes);
    cudaFree(buf->d_sk_bytes);
    cudaFree(buf->d_ct_bytes);
    cudaFree(buf->d_ss_bytes);
    cudaFree(buf->d_coins_kg);
    cudaFree(buf->d_coins_enc);
    cudaFree(buf->d_publicseed_kg);
    cudaFree(buf->d_noiseseed_kg);
}

/* ================================================================
 *  Warp 协同采样 kernel (KEM 密钥生成)
 *  1 warp (32 threads) = 1 实例
 *  Lane 0: SHA3-512 展开种子 → (publicseed, noiseseed)
 *  全部 lanes: 并行展开矩阵 A 和噪声多项式 s, e
 *
 *  输出 SoA:
 *    d_mat[row*K*B*N + col*B*N + inst*N + c] = A[inst][row][col][c]
 *    d_skpv[i*B*N + inst*N + c] = s[inst][i][c]  (未 NTT)
 *    d_e[i*B*N + inst*N + c]    = e[inst][i][c]  (未 NTT)
 * ================================================================ */

#ifndef WP_KG_WARP_SIZE
#define WP_KG_WARP_SIZE  32
#endif

#ifndef WP_KG_WARPS_BLOCK
#define WP_KG_WARPS_BLOCK 4
#endif

#define WP_KG_TPB        (WP_KG_WARP_SIZE * WP_KG_WARPS_BLOCK)

#ifndef KEM_SPLIT_KEYGEN_SAMPLE
#define KEM_SPLIT_KEYGEN_SAMPLE 0
#endif

#ifndef KEM_SERIAL_TPB
#ifdef USE_HIP
#define KEM_SERIAL_TPB 64
#else
#define KEM_SERIAL_TPB 64
#endif
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

__global__ void batch_keygen_warp_sample_kernel(
    int16_t * __restrict__ d_mat,         /* K*K * B * N */
    int16_t * __restrict__ d_skpv,        /* K   * B * N */
    int16_t * __restrict__ d_e,           /* K   * B * N */
    uint8_t * __restrict__ d_publicseed,
    const uint8_t * __restrict__ d_coins, /* B * 2*SYMBYTES */
    int batch_count)
{
    int inst  = blockIdx.x * WP_KG_WARPS_BLOCK + (threadIdx.x / WP_KG_WARP_SIZE);
    int lane  = threadIdx.x & (WP_KG_WARP_SIZE - 1);

    if (inst >= batch_count) return;

    /* Warp-level shared: publicseed 和 noiseseed */
    __shared__ uint8_t ws_pub[WP_KG_WARPS_BLOCK][PARAM_SYMBYTES];
    __shared__ uint8_t ws_noise[WP_KG_WARPS_BLOCK][PARAM_SYMBYTES];

    int warp_id = threadIdx.x / WP_KG_WARP_SIZE;
    uint8_t *publicseed = ws_pub[warp_id];
    uint8_t *noiseseed  = ws_noise[warp_id];

    if (lane == 0) {
        /* 展开种子: SHA3-512(coins[0:32]) → (publicseed[32], noiseseed[32]) */
        uint8_t buf[2 * PARAM_SYMBYTES];
        sha3_512(buf, d_coins + inst * 2 * PARAM_SYMBYTES, PARAM_SYMBYTES);
        for (int i = 0; i < PARAM_SYMBYTES; i++) {
            publicseed[i] = buf[i];
            d_publicseed[(size_t)inst * PARAM_SYMBYTES + i] = buf[i];
        }
        for (int i = 0; i < PARAM_SYMBYTES; i++) noiseseed[i]  = buf[PARAM_SYMBYTES + i];
    }
    __syncwarp();

    /* 矩阵展开: 每个 lane 负责若干多项式 (A[row][col]) */
    int total_mat_polys = PARAM_K * PARAM_K;
    for (int p = lane; p < total_mat_polys; p += WP_KG_WARP_SIZE) {
        int row = p / PARAM_K;
        int col = p % PARAM_K;

        /* 目标地址: SoA 格式 */
        int16_t *dst = d_mat + ((size_t)(row * PARAM_K + col) * batch_count + inst) * PARAM_N;

        uint8_t extseed[PARAM_SYMBYTES + 2];
        for (int i = 0; i < PARAM_SYMBYTES; i++) extseed[i] = publicseed[i];

#if ALGORITHM == ALGO_KYBER
        extseed[PARAM_SYMBYTES]   = (uint8_t)col;  /* j */
        extseed[PARAM_SYMBYTES+1] = (uint8_t)row;  /* i */
#elif ALGORITHM == ALGO_AIGIS_ENC
        extseed[PARAM_SYMBYTES]   = (uint8_t)row;  /* i */
        extseed[PARAM_SYMBYTES+1] = (uint8_t)col;  /* j */
#endif

#if KEM_DIRECT_REJ_UNIFORM
        rej_uniform_xof(dst, publicseed, extseed[PARAM_SYMBYTES], extseed[PARAM_SYMBYTES + 1]);
#else
        keccak_state state;
        shake128_absorb_once(&state, extseed, PARAM_SYMBYTES + 2);

        unsigned int ctr = 0;
        uint8_t buf[PARAM_GEN_MATRIX_NBLOCKS * PARAM_XOF_BLOCKBYTES];
        while (ctr < PARAM_N) {
            shake128_squeezeblocks(buf, PARAM_GEN_MATRIX_NBLOCKS, &state);
            ctr += rej_uniform(dst + ctr, PARAM_N - ctr,
                               buf, PARAM_GEN_MATRIX_NBLOCKS * PARAM_XOF_BLOCKBYTES);
        }
#endif
    }

    /* 噪声采样: s[0..K-1], e[0..K-1] */
    for (int i = lane; i < PARAM_K; i += WP_KG_WARP_SIZE) {
        int16_t *dst_s = d_skpv + ((size_t)i * batch_count + inst) * PARAM_N;
        poly_getnoise_s(dst_s, noiseseed, (uint8_t)i);
    }
    for (int i = lane; i < PARAM_K; i += WP_KG_WARP_SIZE) {
        int16_t *dst_e = d_e + ((size_t)i * batch_count + inst) * PARAM_N;
        poly_getnoise_e_kg(dst_e, noiseseed, (uint8_t)(PARAM_K + i));
    }
}

__global__ void batch_keygen_seed_expand_kernel(
    uint8_t * __restrict__ d_publicseed,
    uint8_t * __restrict__ d_noiseseed,
    const uint8_t * __restrict__ d_coins,
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;

    uint8_t buf[2 * PARAM_SYMBYTES];
    sha3_512(buf, d_coins + (size_t)inst * 2 * PARAM_SYMBYTES, PARAM_SYMBYTES);
    for (int i = 0; i < PARAM_SYMBYTES; i++) {
        d_publicseed[(size_t)inst * PARAM_SYMBYTES + i] = buf[i];
        d_noiseseed[(size_t)inst * PARAM_SYMBYTES + i] = buf[PARAM_SYMBYTES + i];
    }
}

__global__ void batch_keygen_mat_sample_kernel(
    int16_t * __restrict__ d_mat,
    const uint8_t * __restrict__ d_publicseed,
    int batch_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * PARAM_K * PARAM_K;
    if (idx >= total) return;

    int inst = idx % batch_count;
    int p = idx / batch_count;
    int row = p / PARAM_K;
    int col = p % PARAM_K;

#if ALGORITHM == ALGO_KYBER
    uint8_t x = (uint8_t)col;
    uint8_t y = (uint8_t)row;
#elif ALGORITHM == ALGO_AIGIS_ENC
    uint8_t x = (uint8_t)row;
    uint8_t y = (uint8_t)col;
#endif

    int16_t *dst = d_mat + ((size_t)(row * PARAM_K + col) * batch_count + inst) * PARAM_N;
    const uint8_t *seed = d_publicseed + (size_t)inst * PARAM_SYMBYTES;
    rej_uniform_xof(dst, seed, x, y);
}

__global__ void batch_keygen_noise_sample_kernel(
    int16_t * __restrict__ d_skpv,
    int16_t * __restrict__ d_e,
    const uint8_t * __restrict__ d_noiseseed,
    int batch_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * PARAM_K * 2;
    if (idx >= total) return;

    int inst = idx % batch_count;
    int q = idx / batch_count;
    int poly = q % PARAM_K;
    const uint8_t *seed = d_noiseseed + (size_t)inst * PARAM_SYMBYTES;

    if (q < PARAM_K) {
        int16_t *dst = d_skpv + ((size_t)poly * batch_count + inst) * PARAM_N;
        poly_getnoise_s(dst, seed, (uint8_t)poly);
    } else {
        int16_t *dst = d_e + ((size_t)poly * batch_count + inst) * PARAM_N;
        poly_getnoise_e_kg(dst, seed, (uint8_t)(PARAM_K + poly));
    }
}

/* ================================================================
 *  批量打包 PK/SK kernel (每 block 处理一个实例)
 *  在所有 NTT 和 matvec 计算完成后调用
 *
 *  输入:
 *    d_mat       — 矩阵 A (unused for packing, publicseed stored in d_coins)
 *    d_skpv      — NTT 域 s (已 caddq)
 *    d_pkpv      — b = A*s + e (已 caddq), 以 SoA 格式
 *  输出:
 *    d_pk_bytes  — PK 字节流
 *    d_sk_bytes  — SK 字节流 (indcpa_sk || pk || H(pk) || z)
 * ================================================================ */

__global__ void batch_pack_keypair_kernel(
    uint8_t * __restrict__ d_pk_bytes,
    uint8_t * __restrict__ d_sk_bytes,
    const int16_t * __restrict__ d_skpv,
    const int16_t * __restrict__ d_pkpv,
    const uint8_t * __restrict__ d_coins,  /* B * 2*SYMBYTES: publicseed 在位置[inst*2*32] */
    int batch_count)
{
    int inst = blockIdx.x;
    if (inst >= batch_count) return;

    /* 构建 kem_polyvec 结构 (从 SoA 还原为 AoS) */
    kem_polyvec skpv_local, pkpv_local;
    for (int i = 0; i < PARAM_K; i++)
        for (int c = 0; c < PARAM_N; c++) {
            skpv_local.vec[i].coeffs[c] = d_skpv[((size_t)i * batch_count + inst) * PARAM_N + c];
            pkpv_local.vec[i].coeffs[c] = d_pkpv[((size_t)i * batch_count + inst) * PARAM_N + c];
        }

    /* 从 d_coins 取出 publicseed (keygen 时, sha3_512 已展开, publicseed = 前 32 字节) */
    /* 实际上我们在 warp 采样时已用 sha3_512 展开, 这里需要重新计算 publicseed */
    uint8_t seeds[2 * PARAM_SYMBYTES];
    sha3_512(seeds, d_coins + (size_t)inst * 2 * PARAM_SYMBYTES, PARAM_SYMBYTES);
    const uint8_t *publicseed = seeds;

    /* PK = pk_poly_compress(pkpv) || publicseed */
    uint8_t *pk = d_pk_bytes + (size_t)inst * PARAM_PUBLICKEYBYTES;
    pack_pk(pk, &pkpv_local, publicseed);

    /* SK = polyvec_tobytes(skpv) || pk || H(pk) || z */
    uint8_t *sk = d_sk_bytes + (size_t)inst * PARAM_SECRETKEYBYTES;
    pack_sk(sk, &skpv_local);

    /* sk[indcpa_sk_bytes:] = pk */
    for (int i = 0; i < (int)PARAM_PUBLICKEYBYTES; i++)
        sk[PARAM_INDCPA_SECRETKEYBYTES + i] = pk[i];

    /* H(pk) */
    sha3_256(sk + PARAM_INDCPA_SECRETKEYBYTES + PARAM_PUBLICKEYBYTES, pk, PARAM_PUBLICKEYBYTES);

    /* z = coins[32:64] (第二个 32 字节作为随机 z) */
    const uint8_t *z_src = d_coins + (size_t)inst * 2 * PARAM_SYMBYTES + PARAM_SYMBYTES;
    uint8_t *z_dst = sk + PARAM_INDCPA_SECRETKEYBYTES + PARAM_PUBLICKEYBYTES + PARAM_SYMBYTES;
    for (int i = 0; i < PARAM_SYMBYTES; i++) z_dst[i] = z_src[i];
}

#ifndef KEM_PACK_TPB
#define KEM_PACK_TPB 128
#endif

__global__ void batch_pack_sk_polyvec_kernel(
    uint8_t * __restrict__ d_sk_bytes,
    const int16_t * __restrict__ d_skpv,
    int batch_count)
{
    int inst = blockIdx.x;
    int poly = blockIdx.y;
    int tid  = threadIdx.x;
    if (inst >= batch_count || poly >= PARAM_K) return;

    const int16_t *src = d_skpv + ((size_t)poly * batch_count + inst) * PARAM_N;
    uint8_t *out = d_sk_bytes + (size_t)inst * PARAM_SECRETKEYBYTES
                 + (size_t)poly * PARAM_POLYBYTES;

#if ALGORITHM == ALGO_KYBER
    for (int i = tid; i < PARAM_N / 2; i += blockDim.x) {
        int16_t t0 = caddq(src[2 * i]);
        int16_t t1 = caddq(src[2 * i + 1]);
        out[3 * i + 0] = (uint8_t)t0;
        out[3 * i + 1] = (uint8_t)((t0 >> 8) | (t1 << 4));
        out[3 * i + 2] = (uint8_t)(t1 >> 4);
    }
#elif ALGORITHM == ALGO_AIGIS_ENC
    for (int i = tid; i < PARAM_N / 8; i += blockDim.x) {
        int16_t t0 = caddq(src[8 * i + 0]);
        int16_t t1 = caddq(src[8 * i + 1]);
        int16_t t2 = caddq(src[8 * i + 2]);
        int16_t t3 = caddq(src[8 * i + 3]);
        int16_t t4 = caddq(src[8 * i + 4]);
        int16_t t5 = caddq(src[8 * i + 5]);
        int16_t t6 = caddq(src[8 * i + 6]);
        int16_t t7 = caddq(src[8 * i + 7]);
        out[13 * i +  0] = (uint8_t)t0;
        out[13 * i +  1] = (uint8_t)((t0 >> 8) | (t1 << 5));
        out[13 * i +  2] = (uint8_t)(t1 >> 3);
        out[13 * i +  3] = (uint8_t)((t1 >> 11) | (t2 << 2));
        out[13 * i +  4] = (uint8_t)((t2 >> 6) | (t3 << 7));
        out[13 * i +  5] = (uint8_t)(t3 >> 1);
        out[13 * i +  6] = (uint8_t)((t3 >> 9) | (t4 << 4));
        out[13 * i +  7] = (uint8_t)(t4 >> 4);
        out[13 * i +  8] = (uint8_t)((t4 >> 12) | (t5 << 1));
        out[13 * i +  9] = (uint8_t)((t5 >> 7) | (t6 << 6));
        out[13 * i + 10] = (uint8_t)(t6 >> 2);
        out[13 * i + 11] = (uint8_t)((t6 >> 10) | (t7 << 3));
        out[13 * i + 12] = (uint8_t)(t7 >> 5);
    }
#endif
}

__global__ void batch_pack_pk_polyvec_kernel(
    uint8_t * __restrict__ d_pk_bytes,
    const int16_t * __restrict__ d_pkpv,
    int batch_count)
{
    int inst = blockIdx.x;
    int poly = blockIdx.y;
    int tid  = threadIdx.x;
    if (inst >= batch_count || poly >= PARAM_K) return;

    const int16_t *src = d_pkpv + ((size_t)poly * batch_count + inst) * PARAM_N;
    uint8_t *out = d_pk_bytes + (size_t)inst * PARAM_PUBLICKEYBYTES
                 + (size_t)poly * (PARAM_BITS_PK * PARAM_N / 8);

#if ALGORITHM == ALGO_KYBER
    for (int i = tid; i < PARAM_N / 2; i += blockDim.x) {
        int16_t t0 = caddq(src[2 * i]);
        int16_t t1 = caddq(src[2 * i + 1]);
        out[3 * i + 0] = (uint8_t)t0;
        out[3 * i + 1] = (uint8_t)((t0 >> 8) | (t1 << 4));
        out[3 * i + 2] = (uint8_t)(t1 >> 4);
    }
#elif PARAM_BITS_PK == 9
    for (int i = tid; i < PARAM_N / 8; i += blockDim.x) {
        uint16_t c0 = (uint16_t)((((int32_t)caddq(src[8*i+0]) << 9) + PARAM_Q/2) / PARAM_Q) & 0x1FF;
        uint16_t c1 = (uint16_t)((((int32_t)caddq(src[8*i+1]) << 9) + PARAM_Q/2) / PARAM_Q) & 0x1FF;
        uint16_t c2 = (uint16_t)((((int32_t)caddq(src[8*i+2]) << 9) + PARAM_Q/2) / PARAM_Q) & 0x1FF;
        uint16_t c3 = (uint16_t)((((int32_t)caddq(src[8*i+3]) << 9) + PARAM_Q/2) / PARAM_Q) & 0x1FF;
        uint16_t c4 = (uint16_t)((((int32_t)caddq(src[8*i+4]) << 9) + PARAM_Q/2) / PARAM_Q) & 0x1FF;
        uint16_t c5 = (uint16_t)((((int32_t)caddq(src[8*i+5]) << 9) + PARAM_Q/2) / PARAM_Q) & 0x1FF;
        uint16_t c6 = (uint16_t)((((int32_t)caddq(src[8*i+6]) << 9) + PARAM_Q/2) / PARAM_Q) & 0x1FF;
        uint16_t c7 = (uint16_t)((((int32_t)caddq(src[8*i+7]) << 9) + PARAM_Q/2) / PARAM_Q) & 0x1FF;
        out[9*i+0] = (uint8_t)c0;
        out[9*i+1] = (uint8_t)((c0 >> 8) | (c1 << 1));
        out[9*i+2] = (uint8_t)((c1 >> 7) | (c2 << 2));
        out[9*i+3] = (uint8_t)((c2 >> 6) | (c3 << 3));
        out[9*i+4] = (uint8_t)((c3 >> 5) | (c4 << 4));
        out[9*i+5] = (uint8_t)((c4 >> 4) | (c5 << 5));
        out[9*i+6] = (uint8_t)((c5 >> 3) | (c6 << 6));
        out[9*i+7] = (uint8_t)((c6 >> 2) | (c7 << 7));
        out[9*i+8] = (uint8_t)(c7 >> 1);
    }
#elif PARAM_BITS_PK == 10
    for (int i = tid; i < PARAM_N / 4; i += blockDim.x) {
        uint16_t c0 = (uint16_t)((((int32_t)caddq(src[4*i+0]) << 10) + PARAM_Q/2) / PARAM_Q) & 0x3FF;
        uint16_t c1 = (uint16_t)((((int32_t)caddq(src[4*i+1]) << 10) + PARAM_Q/2) / PARAM_Q) & 0x3FF;
        uint16_t c2 = (uint16_t)((((int32_t)caddq(src[4*i+2]) << 10) + PARAM_Q/2) / PARAM_Q) & 0x3FF;
        uint16_t c3 = (uint16_t)((((int32_t)caddq(src[4*i+3]) << 10) + PARAM_Q/2) / PARAM_Q) & 0x3FF;
        out[5*i+0] = (uint8_t)c0;
        out[5*i+1] = (uint8_t)((c0 >> 8) | (c1 << 2));
        out[5*i+2] = (uint8_t)((c1 >> 6) | (c2 << 4));
        out[5*i+3] = (uint8_t)((c2 >> 4) | (c3 << 6));
        out[5*i+4] = (uint8_t)(c3 >> 2);
    }
#elif PARAM_BITS_PK == 11
    for (int i = tid; i < PARAM_N / 8; i += blockDim.x) {
        uint16_t c0 = (uint16_t)((((int32_t)caddq(src[8*i+0]) << 11) + PARAM_Q/2) / PARAM_Q) & 0x7FF;
        uint16_t c1 = (uint16_t)((((int32_t)caddq(src[8*i+1]) << 11) + PARAM_Q/2) / PARAM_Q) & 0x7FF;
        uint16_t c2 = (uint16_t)((((int32_t)caddq(src[8*i+2]) << 11) + PARAM_Q/2) / PARAM_Q) & 0x7FF;
        uint16_t c3 = (uint16_t)((((int32_t)caddq(src[8*i+3]) << 11) + PARAM_Q/2) / PARAM_Q) & 0x7FF;
        uint16_t c4 = (uint16_t)((((int32_t)caddq(src[8*i+4]) << 11) + PARAM_Q/2) / PARAM_Q) & 0x7FF;
        uint16_t c5 = (uint16_t)((((int32_t)caddq(src[8*i+5]) << 11) + PARAM_Q/2) / PARAM_Q) & 0x7FF;
        uint16_t c6 = (uint16_t)((((int32_t)caddq(src[8*i+6]) << 11) + PARAM_Q/2) / PARAM_Q) & 0x7FF;
        uint16_t c7 = (uint16_t)((((int32_t)caddq(src[8*i+7]) << 11) + PARAM_Q/2) / PARAM_Q) & 0x7FF;
        out[11*i+ 0] = (uint8_t)c0;
        out[11*i+ 1] = (uint8_t)((c0 >> 8) | (c1 << 3));
        out[11*i+ 2] = (uint8_t)((c1 >> 5) | (c2 << 6));
        out[11*i+ 3] = (uint8_t)(c2 >> 2);
        out[11*i+ 4] = (uint8_t)((c2 >> 10) | (c3 << 1));
        out[11*i+ 5] = (uint8_t)((c3 >> 7) | (c4 << 4));
        out[11*i+ 6] = (uint8_t)((c4 >> 4) | (c5 << 7));
        out[11*i+ 7] = (uint8_t)(c5 >> 1);
        out[11*i+ 8] = (uint8_t)((c5 >> 9) | (c6 << 2));
        out[11*i+ 9] = (uint8_t)((c6 >> 6) | (c7 << 5));
        out[11*i+10] = (uint8_t)(c7 >> 3);
    }
#endif
}

__global__ void batch_pack_keypair_finalize_kernel(
    uint8_t * __restrict__ d_pk_bytes,
    uint8_t * __restrict__ d_sk_bytes,
    const uint8_t * __restrict__ d_publicseed,
    const uint8_t * __restrict__ d_coins,
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;

    uint8_t *pk = d_pk_bytes + (size_t)inst * PARAM_PUBLICKEYBYTES;
    uint8_t *sk = d_sk_bytes + (size_t)inst * PARAM_SECRETKEYBYTES;
    const uint8_t *rho = d_publicseed + (size_t)inst * PARAM_SYMBYTES;

    for (int i = 0; i < PARAM_SYMBYTES; i++)
        pk[PARAM_PK_POLYVEC_BYTES + i] = rho[i];

    for (int i = 0; i < (int)PARAM_PUBLICKEYBYTES; i++)
        sk[PARAM_INDCPA_SECRETKEYBYTES + i] = pk[i];

    sha3_256(sk + PARAM_INDCPA_SECRETKEYBYTES + PARAM_PUBLICKEYBYTES,
             pk, PARAM_PUBLICKEYBYTES);

    const uint8_t *z_src = d_coins + (size_t)inst * 2 * PARAM_SYMBYTES + PARAM_SYMBYTES;
    uint8_t *z_dst = sk + PARAM_INDCPA_SECRETKEYBYTES + PARAM_PUBLICKEYBYTES + PARAM_SYMBYTES;
    for (int i = 0; i < PARAM_SYMBYTES; i++) z_dst[i] = z_src[i];
}

/* ================================================================
 *  批量单实例 keygen kernel (完整流水线, 单线程 fallback)
 *  用于 batch 较小时, 直接调用 kem_keypair 设备函数
 * ================================================================ */
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

__global__ KEM_KEYPAIR_KERNEL_BOUNDS void batch_kem_keypair_serial_kernel(
    uint8_t * __restrict__ d_pk,
    uint8_t * __restrict__ d_sk,
    const uint8_t * __restrict__ d_coins,  /* B * 2*SYMBYTES */
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;

    kem_keypair(
        d_pk + (size_t)inst * PARAM_PUBLICKEYBYTES,
        d_sk + (size_t)inst * PARAM_SECRETKEYBYTES,
        d_coins + (size_t)inst * 2 * PARAM_SYMBYTES
    );
}

/* ================================================================
 *  批量单实例 encaps kernel
 * ================================================================ */
__global__ KEM_ENCAPS_KERNEL_BOUNDS void batch_kem_encaps_serial_kernel(
    uint8_t * __restrict__ d_ct,
    uint8_t * __restrict__ d_ss,
    const uint8_t * __restrict__ d_pk,
    const uint8_t * __restrict__ d_coins,  /* B * SYMBYTES */
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;

    kem_encaps(
        d_ct + (size_t)inst * PARAM_CIPHERTEXTBYTES,
        d_ss + (size_t)inst * PARAM_SSBYTES,
        d_pk + (size_t)inst * PARAM_PUBLICKEYBYTES,
        d_coins + (size_t)inst * PARAM_SYMBYTES
    );
}

/* ================================================================
 *  批量单实例 decaps kernel
 * ================================================================ */
__global__ KEM_DECAPS_KERNEL_BOUNDS void batch_kem_decaps_serial_kernel(
    uint8_t * __restrict__ d_ss,
    const uint8_t * __restrict__ d_ct,
    const uint8_t * __restrict__ d_sk,
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;

    kem_decaps(
        d_ss + (size_t)inst * PARAM_SSBYTES,
        d_ct + (size_t)inst * PARAM_CIPHERTEXTBYTES,
        d_sk + (size_t)inst * PARAM_SECRETKEYBYTES
    );
}

/* ================================================================
 *  批量 KEM 高性能流水线
 *
 *  batch_keygen_pipelined:
 *    1. Warp 采样 (矩阵 A + s + e)
 *    2. 批量 NTT(s)
 *    3. 2D grid 矩阵向量乘 (A*s → pkpv)
 *    4. 批量 INVNTT(pkpv) + 加 e, caddq
 *    5. 打包 pk/sk
 * ================================================================ */
static inline cudaError_t batch_keygen_pipelined(
    uint8_t *d_pk_out, uint8_t *d_sk_out,
    BatchKemBuffers *buf,
    int batch_count,
    cudaStream_t stream = 0)
{
    cudaError_t err;

    /* 生成随机种子 (device side — 在 host 侧用 cudaMemcpy 传入 d_coins_kg) */

    /* Step 1: Warp 采样 */
    int blocks = (batch_count + WP_KG_WARPS_BLOCK - 1) / WP_KG_WARPS_BLOCK;
#if KEM_SPLIT_KEYGEN_SAMPLE
    batch_keygen_seed_expand_kernel<<<ceil_div(batch_count, KEM_SERIAL_TPB), KEM_SERIAL_TPB, 0, stream>>>(
        buf->d_publicseed_kg, buf->d_noiseseed_kg, buf->d_coins_kg, batch_count);
    batch_keygen_mat_sample_kernel<<<ceil_div(batch_count * PARAM_K * PARAM_K, KEM_SERIAL_TPB), KEM_SERIAL_TPB, 0, stream>>>(
        buf->d_mat, buf->d_publicseed_kg, batch_count);
    batch_keygen_noise_sample_kernel<<<ceil_div(batch_count * PARAM_K * 2, KEM_SERIAL_TPB), KEM_SERIAL_TPB, 0, stream>>>(
        buf->d_skpv, buf->d_e, buf->d_noiseseed_kg, batch_count);
#else
    batch_keygen_warp_sample_kernel<<<blocks, WP_KG_TPB, 0, stream>>>(
        buf->d_mat, buf->d_skpv, buf->d_e,
        buf->d_publicseed_kg, buf->d_coins_kg, batch_count);
#endif

    /* Step 2: 批量 NTT(s) — d_skpv 中 K 个 poly 组 */
    for (int i = 0; i < PARAM_K; i++) {
        int16_t *ptr = buf->d_skpv + (size_t)i * batch_count * PARAM_N;
        batch_ntt_kernel<<<batch_count, 128, 0, stream>>>(ptr, batch_count);
    }

    /* Step 2b: caddq(s) */
    for (int i = 0; i < PARAM_K; i++) {
        int16_t *ptr = buf->d_skpv + (size_t)i * batch_count * PARAM_N;
        launch_batch_caddq(ptr, batch_count, stream);
    }

    /* Step 3: 矩阵向量乘 A * s_hat → pkpv */
    launch_batch_matvec(buf->d_pkpv, buf->d_mat, buf->d_skpv, batch_count, stream);

    /* Step 4: INVNTT(pkpv) */
    for (int i = 0; i < PARAM_K; i++) {
        int16_t *ptr = buf->d_pkpv + (size_t)i * batch_count * PARAM_N;
        batch_invntt_kernel<<<batch_count, 128, 0, stream>>>(ptr, batch_count);
    }

    /* pkpv += e */
    for (int i = 0; i < PARAM_K; i++) {
        launch_batch_add(
            buf->d_pkpv + (size_t)i * batch_count * PARAM_N,
            buf->d_pkpv + (size_t)i * batch_count * PARAM_N,
            buf->d_e    + (size_t)i * batch_count * PARAM_N,
            batch_count, stream);
    }

    /* caddq(pkpv) */
    for (int i = 0; i < PARAM_K; i++) {
        launch_batch_caddq(buf->d_pkpv + (size_t)i * batch_count * PARAM_N, batch_count, stream);
    }

    /* Step 5: 打包 PK/SK */
    dim3 pack_grid(batch_count, PARAM_K);
    batch_pack_sk_polyvec_kernel<<<pack_grid, KEM_PACK_TPB, 0, stream>>>(
        d_sk_out, buf->d_skpv, batch_count);
    batch_pack_pk_polyvec_kernel<<<pack_grid, KEM_PACK_TPB, 0, stream>>>(
        d_pk_out, buf->d_pkpv, batch_count);
    batch_pack_keypair_finalize_kernel<<<ceil_div(batch_count, KEM_SERIAL_TPB), KEM_SERIAL_TPB, 0, stream>>>(
        d_pk_out, d_sk_out, buf->d_publicseed_kg, buf->d_coins_kg, batch_count);

    err = cudaGetLastError();
    return err;
}

/* ================================================================
 *  简化批量 encaps/decaps (串行 kernel, 可进一步并行化)
 * ================================================================ */
static inline cudaError_t batch_keygen_pipelined_profile(
    uint8_t *d_pk_out, uint8_t *d_sk_out,
    BatchKemBuffers *buf,
    int batch_count,
    cudaStream_t stream = 0)
{
    cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5, ev6;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);
    cudaEventCreate(&ev3); cudaEventCreate(&ev4); cudaEventCreate(&ev5); cudaEventCreate(&ev6);

    cudaEventRecord(ev0, stream);
    int blocks = (batch_count + WP_KG_WARPS_BLOCK - 1) / WP_KG_WARPS_BLOCK;
#if KEM_SPLIT_KEYGEN_SAMPLE
    batch_keygen_seed_expand_kernel<<<ceil_div(batch_count, KEM_SERIAL_TPB), KEM_SERIAL_TPB, 0, stream>>>(
        buf->d_publicseed_kg, buf->d_noiseseed_kg, buf->d_coins_kg, batch_count);
    batch_keygen_mat_sample_kernel<<<ceil_div(batch_count * PARAM_K * PARAM_K, KEM_SERIAL_TPB), KEM_SERIAL_TPB, 0, stream>>>(
        buf->d_mat, buf->d_publicseed_kg, batch_count);
    batch_keygen_noise_sample_kernel<<<ceil_div(batch_count * PARAM_K * 2, KEM_SERIAL_TPB), KEM_SERIAL_TPB, 0, stream>>>(
        buf->d_skpv, buf->d_e, buf->d_noiseseed_kg, batch_count);
#else
    batch_keygen_warp_sample_kernel<<<blocks, WP_KG_TPB, 0, stream>>>(
        buf->d_mat, buf->d_skpv, buf->d_e,
        buf->d_publicseed_kg, buf->d_coins_kg, batch_count);
#endif
    cudaEventRecord(ev1, stream);

    for (int i = 0; i < PARAM_K; i++) {
        int16_t *ptr = buf->d_skpv + (size_t)i * batch_count * PARAM_N;
        batch_ntt_kernel<<<batch_count, 128, 0, stream>>>(ptr, batch_count);
    }
    for (int i = 0; i < PARAM_K; i++) {
        int16_t *ptr = buf->d_skpv + (size_t)i * batch_count * PARAM_N;
        launch_batch_caddq(ptr, batch_count, stream);
    }
    cudaEventRecord(ev2, stream);

    launch_batch_matvec(buf->d_pkpv, buf->d_mat, buf->d_skpv, batch_count, stream);
    cudaEventRecord(ev3, stream);

    for (int i = 0; i < PARAM_K; i++) {
        int16_t *ptr = buf->d_pkpv + (size_t)i * batch_count * PARAM_N;
        batch_invntt_kernel<<<batch_count, 128, 0, stream>>>(ptr, batch_count);
    }
    cudaEventRecord(ev4, stream);

    for (int i = 0; i < PARAM_K; i++) {
        launch_batch_add(
            buf->d_pkpv + (size_t)i * batch_count * PARAM_N,
            buf->d_pkpv + (size_t)i * batch_count * PARAM_N,
            buf->d_e    + (size_t)i * batch_count * PARAM_N,
            batch_count, stream);
    }
    for (int i = 0; i < PARAM_K; i++)
        launch_batch_caddq(buf->d_pkpv + (size_t)i * batch_count * PARAM_N, batch_count, stream);
    cudaEventRecord(ev5, stream);

    dim3 pack_grid(batch_count, PARAM_K);
    batch_pack_sk_polyvec_kernel<<<pack_grid, KEM_PACK_TPB, 0, stream>>>(
        d_sk_out, buf->d_skpv, batch_count);
    batch_pack_pk_polyvec_kernel<<<pack_grid, KEM_PACK_TPB, 0, stream>>>(
        d_pk_out, buf->d_pkpv, batch_count);
    batch_pack_keypair_finalize_kernel<<<ceil_div(batch_count, KEM_SERIAL_TPB), KEM_SERIAL_TPB, 0, stream>>>(
        d_pk_out, d_sk_out, buf->d_publicseed_kg, buf->d_coins_kg, batch_count);
    cudaEventRecord(ev6, stream);
    cudaEventSynchronize(ev6);

    float sample_ms, ntt_ms, matvec_ms, invntt_ms, add_ms, pack_ms, total_ms;
    cudaEventElapsedTime(&sample_ms, ev0, ev1);
    cudaEventElapsedTime(&ntt_ms,    ev1, ev2);
    cudaEventElapsedTime(&matvec_ms, ev2, ev3);
    cudaEventElapsedTime(&invntt_ms, ev3, ev4);
    cudaEventElapsedTime(&add_ms,    ev4, ev5);
    cudaEventElapsedTime(&pack_ms,   ev5, ev6);
    cudaEventElapsedTime(&total_ms,  ev0, ev6);
    printf("  Pipeline profile: sample=%.3f ntt=%.3f matvec=%.3f invntt=%.3f add=%.3f pack=%.3f total=%.3f ms\n",
           sample_ms, ntt_ms, matvec_ms, invntt_ms, add_ms, pack_ms, total_ms);

    cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2);
    cudaEventDestroy(ev3); cudaEventDestroy(ev4); cudaEventDestroy(ev5); cudaEventDestroy(ev6);
    return cudaGetLastError();
}

static inline cudaError_t batch_encaps_serial(
    uint8_t *d_ct, uint8_t *d_ss,
    const uint8_t *d_pk,
    BatchKemBuffers *buf,
    int batch_count,
    cudaStream_t stream = 0)
{
    int tpb = KEM_ENCAPS_TPB;
    int blocks = (batch_count + tpb - 1) / tpb;
    batch_kem_encaps_serial_kernel<<<blocks, tpb, 0, stream>>>(
        d_ct, d_ss, d_pk, buf->d_coins_enc, batch_count);
    return cudaGetLastError();
}

static inline cudaError_t batch_decaps_serial(
    uint8_t *d_ss,
    const uint8_t *d_ct, const uint8_t *d_sk,
    int batch_count,
    cudaStream_t stream = 0)
{
    int tpb = KEM_DECAPS_TPB;
    int blocks = (batch_count + tpb - 1) / tpb;
    batch_kem_decaps_serial_kernel<<<blocks, tpb, 0, stream>>>(
        d_ss, d_ct, d_sk, batch_count);
    return cudaGetLastError();
}

#endif /* BATCH_KEM_CUH */
