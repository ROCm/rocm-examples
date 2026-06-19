#include "hip/hip_runtime.h"
/*
 * batch_verify.cuh — 分解式批量验证 pipeline
 *
 * 核心优化:
 *   1. 矩阵 A 和 t1_hat 只计算一次, 所有 instance 共享 (precompute)
 *   2. NTT(z) 使用 shared-memory 批量 kernel
 *   3. 矩阵向量乘使用 2D grid (batch × K)
 *   4. SoA 内存布局: z[poly_idx][inst][coeff]
 *   5. 逐系数操作使用 256 threads/block 批量 kernel
 *
 * Pipeline:
 *   [0] 预计算: pk → A(NTT域), t1<<D(NTT域), tr=H(pk)    (一次性)
 *   [1] 解包签名 → z, h, challenge 数据                    (#if 分叉)
 *   [2] 范数检查 z
 *   [3] NTT(z)                                              (shared-mem)
 *   [4] w = A · z_hat                                       (2D grid)
 *   [5] 生成挑战多项式 cp                                   (#if 分叉)
 *   [6] NTT(cp)
 *   [7] w -= cp · t1_hat                                    (2D grid)
 *   [8] reduce + INVNTT + normalize
 *   [9] w1 = use_hint(w, h)
 *  [10] mu = H(tr || [pre ||] m)                            (#if 分叉)
 *  [11] 最终比较: H(mu || pack(w1)) vs 原始挑战              (#if 分叉)
 */

#ifndef BATCH_VERIFY_CUH
#define BATCH_VERIFY_CUH

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>
#include "params.h"
#include "reduce.cuh"
#include "ntt.cuh"
#include "rounding.cuh"
#include "fips202.cuh"
#include "batch_ntt.cuh"
#include "batch_ops.cuh"
#include "poly.cuh"
#include "polyvec.cuh"
#include "packing.cuh"
#include "sign.cuh"
#include "symmetric.cuh"

/* ================================================================
 * 缓冲区结构体
 * ================================================================ */
struct BatchVerifyBuffers {
    /* 预计算共享材料 (来自公钥, 只计算一次) */
    coeff_t *d_mat;           /* K * L * N — 矩阵 A (NTT 域) */
    coeff_t *d_t1_hat;        /* K * N — t1 << D (NTT 域) */
    unsigned char *d_tr;      /* TRBYTES — H(pk) */

    /* 每批次工作缓冲区 — SoA: [poly_idx][batch][coeff] */
    coeff_t *d_z;             /* L * B * N */
    coeff_t *d_h;             /* K * B * N */
    coeff_t *d_cp;            /* B * N — 挑战多项式 */
    coeff_t *d_w;             /* K * B * N */
    coeff_t *d_w1;            /* K * B * N */
    unsigned char *d_mu;      /* B * CRHBYTES */
    int *d_results;           /* B */
    unsigned char *d_raw_sigs;/* B * CRYPTO_BYTES */

    /* 挑战原始数据:
     *   Aigis:  B * N * sizeof(coeff_t) — 完整挑战多项式
     *   ML-DSA: B * CTILDEBYTES — 挑战种子 */
    unsigned char *d_cbuf;

    int max_batch;
};

/* ================================================================
 * 预计算 kernel — 从公钥提取矩阵 A 和 t1_hat
 * ================================================================ */
__global__ void batch_verify_precompute_kernel(
    coeff_t * __restrict__ d_mat,
    coeff_t * __restrict__ d_t1_hat,
    unsigned char * __restrict__ d_tr,
    const unsigned char * __restrict__ pk)
{
    unsigned char rho[SEEDBYTES];
    polyveck t1;
    polyvecl mat[PARAM_K];

    unpack_pk(rho, &t1, pk);
    polyvec_matrix_expand(mat, rho);

    /* 存储矩阵 A 到 flat 数组 */
    for (int i = 0; i < PARAM_K; i++)
        for (int j = 0; j < PARAM_L; j++)
            for (int c = 0; c < PARAM_N; c++)
                d_mat[(i * PARAM_L + j) * PARAM_N + c]
                    = mat[i].vec[j].coeffs[c];

    /* t1 << D, 然后 NTT */
    for (int i = 0; i < PARAM_K; i++) {
        for (int c = 0; c < PARAM_N; c++)
            d_t1_hat[i * PARAM_N + c] = t1.vec[i].coeffs[c] << PARAM_D;
        ntt(d_t1_hat + i * PARAM_N);
    }

    /* tr = H(pk) */
    shake256(d_tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
}

/* ================================================================
 * 解包签名 kernel — SoA 输出
 *
 * 输出: d_z[poly][inst][coeff], d_h[poly][inst][coeff], d_cbuf
 * ================================================================ */

#if ALGORITHM == ALGO_MLDSA

__global__ void __launch_bounds__(64)
batch_verify_unpack_kernel(
    coeff_t * __restrict__ d_z,
    coeff_t * __restrict__ d_h,
    unsigned char * __restrict__ d_cbuf,
    int * __restrict__ d_results,
    const unsigned char * __restrict__ d_raw_sigs,
    int batch_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    const uint8_t *sig = d_raw_sigs + (size_t)idx * CRYPTO_BYTES;
    uint8_t *c_out = d_cbuf + (size_t)idx * CTILDEBYTES;

    /* 解包 ctilde */
    for (int i = 0; i < CTILDEBYTES; i++)
        c_out[i] = sig[i];
    const uint8_t *sp = sig + CTILDEBYTES;

    /* 解包 z → SoA */
#if PARAM_GAMMA1 == (1 << 17)
    for (int l = 0; l < PARAM_L; l++) {
        const uint8_t *a = sp + l * POLYZ_PACKEDBYTES;
        coeff_t *r = d_z + (size_t)l * batch_count * PARAM_N + (size_t)idx * PARAM_N;
        for (unsigned int i = 0; i < PARAM_N/4; i++) {
            r[4*i+0]  = a[9*i+0];
            r[4*i+0] |= (uint32_t)a[9*i+1] << 8;
            r[4*i+0] |= (uint32_t)a[9*i+2] << 16;
            r[4*i+0] &= 0x3FFFF;
            r[4*i+1]  = a[9*i+2] >> 2;
            r[4*i+1] |= (uint32_t)a[9*i+3] << 6;
            r[4*i+1] |= (uint32_t)a[9*i+4] << 14;
            r[4*i+1] &= 0x3FFFF;
            r[4*i+2]  = a[9*i+4] >> 4;
            r[4*i+2] |= (uint32_t)a[9*i+5] << 4;
            r[4*i+2] |= (uint32_t)a[9*i+6] << 12;
            r[4*i+2] &= 0x3FFFF;
            r[4*i+3]  = a[9*i+6] >> 6;
            r[4*i+3] |= (uint32_t)a[9*i+7] << 2;
            r[4*i+3] |= (uint32_t)a[9*i+8] << 10;
            r[4*i+3] &= 0x3FFFF;
            r[4*i+0] = PARAM_GAMMA1 - r[4*i+0];
            r[4*i+1] = PARAM_GAMMA1 - r[4*i+1];
            r[4*i+2] = PARAM_GAMMA1 - r[4*i+2];
            r[4*i+3] = PARAM_GAMMA1 - r[4*i+3];
        }
    }
#elif PARAM_GAMMA1 == (1 << 19)
    for (int l = 0; l < PARAM_L; l++) {
        const uint8_t *a = sp + l * POLYZ_PACKEDBYTES;
        coeff_t *r = d_z + (size_t)l * batch_count * PARAM_N + (size_t)idx * PARAM_N;
        for (unsigned int i = 0; i < PARAM_N/2; i++) {
            r[2*i+0]  = a[5*i+0];
            r[2*i+0] |= (uint32_t)a[5*i+1] << 8;
            r[2*i+0] |= (uint32_t)a[5*i+2] << 16;
            r[2*i+0] &= 0xFFFFF;
            r[2*i+1]  = a[5*i+2] >> 4;
            r[2*i+1] |= (uint32_t)a[5*i+3] << 4;
            r[2*i+1] |= (uint32_t)a[5*i+4] << 12;
            r[2*i+0] = PARAM_GAMMA1 - r[2*i+0];
            r[2*i+1] = PARAM_GAMMA1 - r[2*i+1];
        }
    }
#endif
    sp += PARAM_L * POLYZ_PACKEDBYTES;

    /* 解包 hint → SoA */
    unsigned int k = 0;
    int valid = 1;
    for (int i = 0; i < PARAM_K; i++) {
        if (sp[PARAM_OMEGA + i] < k || sp[PARAM_OMEGA + i] > PARAM_OMEGA) { valid = 0; break; }
        for (unsigned int j = k; j < sp[PARAM_OMEGA + i]; j++) {
            if (j > k && sp[j] <= sp[j-1]) { valid = 0; break; }
            d_h[(size_t)i * batch_count * PARAM_N + (size_t)idx * PARAM_N + sp[j]] = 1;
        }
        if (!valid) break;
        k = sp[PARAM_OMEGA + i];
    }
    if (valid) {
        for (unsigned int j = k; j < PARAM_OMEGA; j++)
            if (sp[j]) { valid = 0; break; }
    }
    if (!valid) d_results[idx] = -1;
}

#elif ALGORITHM == ALGO_AIGIS

__global__ void __launch_bounds__(64)
batch_verify_unpack_kernel(
    coeff_t * __restrict__ d_z,
    coeff_t * __restrict__ d_h,
    unsigned char * __restrict__ d_cbuf,
    int * __restrict__ d_results,
    const unsigned char * __restrict__ d_raw_sigs,
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;

    const unsigned char *sig = d_raw_sigs + (size_t)inst * CRYPTO_BYTES;

    /* 解包 z → SoA (Aigis: 18-bit GAMMA1=2^17) */
    for (unsigned int i = 0; i < PARAM_L; i++) {
        const unsigned char *src = sig + i * POLYZ_PACKEDBYTES;
        coeff_t *dst = d_z + (size_t)i * batch_count * PARAM_N + (size_t)inst * PARAM_N;
        for (unsigned int ii = 0; ii < PARAM_N / 4; ++ii) {
            const unsigned char *a = src + 9 * ii;
            int32_t r0, r1, r2, r3;
            r0  = a[0]; r0 |= (uint32_t)a[1] << 8; r0 |= (uint32_t)(a[2] & 0x03) << 16;
            r0 = PARAM_GAMMA1 - 1 - r0; r0 += ((int32_t)r0 >> 31) & PARAM_Q;
            r1  = a[2] >> 2; r1 |= (uint32_t)a[3] << 6; r1 |= (uint32_t)(a[4] & 0x0F) << 14;
            r1 = PARAM_GAMMA1 - 1 - r1; r1 += ((int32_t)r1 >> 31) & PARAM_Q;
            r2  = a[4] >> 4; r2 |= (uint32_t)a[5] << 4; r2 |= (uint32_t)(a[6] & 0x3F) << 12;
            r2 = PARAM_GAMMA1 - 1 - r2; r2 += ((int32_t)r2 >> 31) & PARAM_Q;
            r3  = a[6] >> 6; r3 |= (uint32_t)a[7] << 2; r3 |= (uint32_t)a[8] << 10;
            r3 = PARAM_GAMMA1 - 1 - r3; r3 += ((int32_t)r3 >> 31) & PARAM_Q;
            dst[4*ii+0] = r0; dst[4*ii+1] = r1; dst[4*ii+2] = r2; dst[4*ii+3] = r3;
        }
    }
    sig += PARAM_L * POLYZ_PACKEDBYTES;

    /* 解包 hint → SoA */
    unsigned int k = 0;
    int fail = 0;
    for (unsigned int i = 0; i < PARAM_K; i++) {
        coeff_t *hdst = d_h + (size_t)i * batch_count * PARAM_N + (size_t)inst * PARAM_N;
        if (sig[PARAM_OMEGA + i] < k || sig[PARAM_OMEGA + i] > PARAM_OMEGA) { fail = 1; break; }
        for (unsigned int j = k; j < sig[PARAM_OMEGA + i]; j++) {
            if (j > k && sig[j] <= sig[j - 1]) { fail = 1; break; }
            hdst[sig[j]] = 1;
        }
        if (fail) break;
        k = sig[PARAM_OMEGA + i];
    }
    if (!fail) {
        for (unsigned int j = k; j < PARAM_OMEGA; j++)
            if (sig[j]) { fail = 1; break; }
    }
    sig += PARAM_OMEGA + PARAM_K;

    /* 解包挑战多项式 (Aigis: 位图 + 符号) */
    coeff_t *cdst = (coeff_t *)(d_cbuf + (size_t)inst * PARAM_N * sizeof(coeff_t));
    for (unsigned int i = 0; i < PARAM_N; i++) cdst[i] = 0;
    if (!fail) {
        uint64_t signs = 0;
        for (unsigned int i = 0; i < 8; i++)
            signs |= (uint64_t)sig[PARAM_N / 8 + i] << (8 * i);
        uint64_t mask = 1;
        for (unsigned int i = 0; i < PARAM_N / 8; i++) {
            for (unsigned int j = 0; j < 8; j++) {
                if ((sig[i] >> j) & 0x01) {
                    cdst[8 * i + j] = (signs & mask) ? PARAM_Q - 1 : 1;
                    mask <<= 1;
                }
            }
        }
    }

    d_results[inst] = fail ? -1 : 0;
}

#endif /* ALGORITHM unpack kernel */

/* ================================================================
 * 范数检查 z kernel — 共用
 * ================================================================ */
__global__ void batch_verify_chknorm_z_kernel(
    int * __restrict__ d_results,
    const coeff_t * __restrict__ d_z,
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;
    if (d_results[inst] != 0) return;

#if ALGORITHM == ALGO_AIGIS
    const int32_t bound = PARAM_GAMMA1 - PARAM_BETA1;
    for (unsigned int i = 0; i < PARAM_L; i++) {
        const coeff_t *zp = d_z + (size_t)i * batch_count * PARAM_N + (size_t)inst * PARAM_N;
        for (unsigned int j = 0; j < PARAM_N; j++) {
            int32_t t = (PARAM_Q - 1) / 2 - (int32_t)zp[j];
            t ^= (t >> 31);
            t = (PARAM_Q - 1) / 2 - t;
            if (t >= bound) { d_results[inst] = -1; return; }
        }
    }
#else /* ALGO_MLDSA */
    const int32_t bound = PARAM_GAMMA1 - PARAM_BETA1;
    for (int l = 0; l < PARAM_L; l++) {
        const coeff_t *zp = d_z + (size_t)l * batch_count * PARAM_N + (size_t)inst * PARAM_N;
        for (int j = 0; j < PARAM_N; j++) {
            int32_t t = zp[j];
            int32_t mask = t >> 31;
            t = t - (mask & 2*t);
            if (t >= bound) { d_results[inst] = -1; return; }
        }
    }
#endif
}

/* ================================================================
 * 矩阵向量乘 kernel — 共用 (共享矩阵 A)
 *
 * w[row] = Σ_{col} A[row][col] · z_hat[col]
 * 注意: A 不含 batch 维度, z 是 SoA 布局
 * grid: (batch_count, PARAM_K)
 * block: PARAM_N threads
 * ================================================================ */
__global__ void batch_verify_matvec_kernel(
    coeff_t * __restrict__ d_w,
    const coeff_t * __restrict__ d_mat,
    const coeff_t * __restrict__ d_z_ntt,
    int batch_count)
{
    int inst = blockIdx.x;
    int row  = blockIdx.y;
    if (inst >= batch_count) return;
    int tid = threadIdx.x;

    coeff_t acc = 0;
    for (int col = 0; col < PARAM_L; col++) {
        /* A 共享: d_mat[(row * L + col) * N + tid] */
        coeff_t a = d_mat[(row * PARAM_L + col) * PARAM_N + tid];
        /* z SoA: d_z_ntt[col * B * N + inst * N + tid] */
        coeff_t b = d_z_ntt[(size_t)col * batch_count * PARAM_N
                            + (size_t)inst * PARAM_N + tid];
        acc += (coeff_t)montgomery_reduce((coeff2_t)a * b);
    }
    /* w SoA output: [row * B * N + inst * N + tid] */
    d_w[(size_t)row * batch_count * PARAM_N + (size_t)inst * PARAM_N + tid] = coeff_reduce(acc);
}

/* ================================================================
 * 融合核: w -= cp · t1_hat  (2D grid)
 * ================================================================ */
__global__ void batch_verify_sub_cp_t1_kernel(
    coeff_t * __restrict__ d_w,
    const coeff_t * __restrict__ d_cp,
    const coeff_t * __restrict__ d_t1_hat,
    int batch_count)
{
    int inst = blockIdx.x;
    int k    = blockIdx.y;
    if (inst >= batch_count) return;
    int tid = threadIdx.x;

    coeff_t c = d_cp[(size_t)inst * PARAM_N + tid];
    coeff_t t = d_t1_hat[k * PARAM_N + tid];
    coeff_t prod = (coeff_t)montgomery_reduce((coeff2_t)c * t);
    size_t idx = (size_t)k * batch_count * PARAM_N + (size_t)inst * PARAM_N + tid;
    d_w[idx] = coeff_sub(d_w[idx], prod);
}

/* ================================================================
 * 挑战多项式生成 kernel
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

__global__ void __launch_bounds__(64)
batch_verify_challenge_kernel(
    coeff_t * __restrict__ d_cp,
    const unsigned char * __restrict__ d_cbuf,
    int batch_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    const uint8_t *c_seed = d_cbuf + (size_t)idx * CTILDEBYTES;
    coeff_t *cp = d_cp + (size_t)idx * PARAM_N;

    unsigned int i, b, pos;
    uint64_t signs;
    uint8_t buf[SHAKE256_RATE];
    keccak_state state;

    shake256_init(&state);
    shake256_absorb(&state, c_seed, CTILDEBYTES);
    shake256_finalize(&state);
    shake256_squeezeblocks(buf, 1, &state);

    signs = 0;
    for (i = 0; i < 8; i++)
        signs |= (uint64_t)buf[i] << (8*i);
    pos = 8;

    for (i = 0; i < PARAM_N; i++) cp[i] = 0;
    for (i = PARAM_N - PARAM_TAU; i < PARAM_N; i++) {
        do {
            if (pos >= SHAKE256_RATE) {
                shake256_squeezeblocks(buf, 1, &state);
                pos = 0;
            }
            b = buf[pos++];
        } while (b > i);
        cp[i] = cp[b];
        cp[b] = 1 - 2*(signs & 1);
        signs >>= 1;
    }
}

#endif /* ALGO_MLDSA challenge kernel */

/* ================================================================
 * mu 计算 kernel
 * ================================================================ */
__global__ void batch_verify_compute_mu_kernel(
    unsigned char * __restrict__ d_mu,
    const unsigned char * __restrict__ d_tr,
    const unsigned char * __restrict__ d_msgs,
    size_t mlen,
    const unsigned char * __restrict__ d_pre,
    size_t prelen,
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;

#if ALGORITHM == ALGO_AIGIS
    /* Aigis: mu = shake256(tr || m_i) */
    keccak_state state;
    shake256_init(&state);
    shake256_absorb(&state, d_tr, CRHBYTES);
    shake256_absorb(&state, d_msgs + (size_t)inst * mlen, mlen);
    shake256_finalize(&state);
    shake256_squeeze(d_mu + (size_t)inst * CRHBYTES, CRHBYTES, &state);
#else
    /* ML-DSA: mu = shake256(tr || pre || m) */
    keccak_state state;
    shake256_init(&state);
    shake256_absorb(&state, d_tr, TRBYTES);
    shake256_absorb(&state, d_pre, prelen);
    shake256_absorb(&state, d_msgs + (size_t)inst * mlen, mlen);
    shake256_finalize(&state);
    shake256_squeeze(d_mu + (size_t)inst * CRHBYTES, CRHBYTES, &state);
#endif
}

/* ================================================================
 * 最终比较 kernel
 * ================================================================ */
#if ALGORITHM == ALGO_AIGIS

__global__ void __launch_bounds__(32)
batch_verify_compare_kernel(
    int * __restrict__ d_results,
    const unsigned char * __restrict__ d_mu,
    const coeff_t * __restrict__ d_w1,
    const unsigned char * __restrict__ d_cbuf,
    int batch_count)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= batch_count) return;
    if (d_results[inst] != 0) return;

    /* 打包 w1 并构造 hash 输入 */
    unsigned char inbuf[CRHBYTES + PARAM_K * POLYW1_PACKEDBYTES];
    const unsigned char *my_mu = d_mu + (size_t)inst * CRHBYTES;
    for (unsigned int i = 0; i < CRHBYTES; i++) inbuf[i] = my_mu[i];

    for (unsigned int ki = 0; ki < PARAM_K; ki++) {
        const coeff_t *w1_poly = d_w1 + (size_t)ki * batch_count * PARAM_N
                                       + (size_t)inst * PARAM_N;
        unsigned char *r = inbuf + CRHBYTES + ki * POLYW1_PACKEDBYTES;
        /* Aigis w1 packing: 3 bits per coeff, 8 coeffs per 3 bytes */
        for (unsigned int i = 0; i < PARAM_N / 8; i++) {
            r[3*i+0] =  w1_poly[8*i+0]       | (w1_poly[8*i+1] << 3) | (w1_poly[8*i+2] << 6);
            r[3*i+1] = (w1_poly[8*i+2] >> 2) | (w1_poly[8*i+3] << 1)
                      | (w1_poly[8*i+4] << 4) | (w1_poly[8*i+5] << 7);
            r[3*i+2] = (w1_poly[8*i+5] >> 1) | (w1_poly[8*i+6] << 2) | (w1_poly[8*i+7] << 5);
        }
    }

    /* 重算挑战多项式: SampleInBall(H(mu || w1_packed)) */
    unsigned char outbuf[SHAKE256_RATE];
    keccak_state state;
    shake256_absorb_once(&state, inbuf, CRHBYTES + PARAM_K * POLYW1_PACKEDBYTES);
    shake256_squeezeblocks(outbuf, 1, &state);

    uint64_t signs = 0;
    for (unsigned int i = 0; i < 8; i++) signs |= (uint64_t)outbuf[i] << (8 * i);
    unsigned int pos = 8;
    uint64_t mask = 1;

    coeff_t cp[PARAM_N];
    for (unsigned int i = 0; i < PARAM_N; i++) cp[i] = 0;

    for (unsigned int i = 196; i < 256; i++) {
        unsigned int b;
        do {
            if (pos >= SHAKE256_RATE) {
                shake256_squeezeblocks(outbuf, 1, &state);
                pos = 0;
            }
            b = outbuf[pos++];
        } while (b > i);
        cp[i] = cp[b];
        cp[b] = (signs & mask) ? PARAM_Q - 1 : 1;
        mask <<= 1;
    }

    /* 与解包的原始挑战多项式逐系数比较 */
    const coeff_t *c_orig = (const coeff_t *)(d_cbuf + (size_t)inst * PARAM_N * sizeof(coeff_t));
    for (unsigned int i = 0; i < PARAM_N; i++) {
        if (c_orig[i] != cp[i]) {
            d_results[inst] = -1; return;
        }
    }
}

#elif ALGORITHM == ALGO_MLDSA

__global__ void __launch_bounds__(128)
batch_verify_compare_kernel(
    int * __restrict__ d_results,
    const unsigned char * __restrict__ d_mu,
    const coeff_t * __restrict__ d_w1,
    const unsigned char * __restrict__ d_cbuf,
    int batch_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;
    if (d_results[idx] != 0) return;  /* 已失败的实例跳过 */

    const uint8_t *c_orig = d_cbuf + (size_t)idx * CTILDEBYTES;

    keccak_state state;
    shake256_init(&state);
    shake256_absorb(&state, d_mu + (size_t)idx * CRHBYTES, CRHBYTES);

    uint8_t w1_pack[POLYW1_PACKEDBYTES];
    for (int k = 0; k < PARAM_K; k++) {
        const coeff_t *w1k = d_w1 + (size_t)k * batch_count * PARAM_N + (size_t)idx * PARAM_N;
#if PARAM_GAMMA2 == (PARAM_Q-1)/88
        for (unsigned int i = 0; i < PARAM_N/4; i++) {
            w1_pack[3*i+0]  = w1k[4*i+0];
            w1_pack[3*i+0] |= w1k[4*i+1] << 6;
            w1_pack[3*i+1]  = w1k[4*i+1] >> 2;
            w1_pack[3*i+1] |= w1k[4*i+2] << 4;
            w1_pack[3*i+2]  = w1k[4*i+2] >> 4;
            w1_pack[3*i+2] |= w1k[4*i+3] << 2;
        }
#elif PARAM_GAMMA2 == (PARAM_Q-1)/32
        for (unsigned int i = 0; i < PARAM_N/2; i++)
            w1_pack[i] = w1k[2*i+0] | (w1k[2*i+1] << 4);
#endif
        shake256_absorb(&state, w1_pack, POLYW1_PACKEDBYTES);
    }

    shake256_finalize(&state);

    uint8_t c2[CTILDEBYTES];
    shake256_squeeze(c2, CTILDEBYTES, &state);

    int result = 0;
    for (unsigned int i = 0; i < CTILDEBYTES; i++) {
        if (c_orig[i] != c2[i]) { result = -1; break; }
    }
    d_results[idx] = result;
}

#endif /* ALGORITHM compare kernel */

/* ================================================================
 * Host API — 缓冲区分配/释放
 * ================================================================ */

static int batch_verify_alloc(BatchVerifyBuffers *buf, int max_batch) {
    memset(buf, 0, sizeof(*buf));
    buf->max_batch = max_batch;
    size_t B = max_batch;
    size_t N = PARAM_N;

#define BV_TRY(ptr, sz) do { \
    if (hipMalloc(&(ptr), (sz)) != hipSuccess) { hipGetLastError(); return -1; } \
} while(0)

    BV_TRY(buf->d_mat,      PARAM_K * PARAM_L * N * sizeof(coeff_t));
    BV_TRY(buf->d_t1_hat,   PARAM_K * N * sizeof(coeff_t));
    BV_TRY(buf->d_tr,       TRBYTES);
    BV_TRY(buf->d_z,        PARAM_L * B * N * sizeof(coeff_t));
    BV_TRY(buf->d_h,        PARAM_K * B * N * sizeof(coeff_t));
    BV_TRY(buf->d_cp,       B * N * sizeof(coeff_t));
    BV_TRY(buf->d_w,        PARAM_K * B * N * sizeof(coeff_t));
    BV_TRY(buf->d_w1,       PARAM_K * B * N * sizeof(coeff_t));
    BV_TRY(buf->d_mu,       B * CRHBYTES);
    BV_TRY(buf->d_results,  B * sizeof(int));
    BV_TRY(buf->d_raw_sigs, B * CRYPTO_BYTES);

#if ALGORITHM == ALGO_AIGIS
    BV_TRY(buf->d_cbuf,     B * N * sizeof(coeff_t));
#else
    BV_TRY(buf->d_cbuf,     B * CTILDEBYTES);
#endif

#undef BV_TRY
    return 0;
}

static void batch_verify_free(BatchVerifyBuffers *buf) {
    hipFree(buf->d_mat);    hipFree(buf->d_t1_hat);
    hipFree(buf->d_tr);     hipFree(buf->d_z);
    hipFree(buf->d_h);      hipFree(buf->d_cp);
    hipFree(buf->d_w);      hipFree(buf->d_w1);
    hipFree(buf->d_mu);     hipFree(buf->d_results);
    hipFree(buf->d_raw_sigs); hipFree(buf->d_cbuf);
    memset(buf, 0, sizeof(*buf));
}

/* ================================================================
 * 批量验证核心 pipeline
 *
 * 前置条件: buf->d_raw_sigs 已经准备好 (device 端 AoS 签名数据)
 * ================================================================ */
static int batch_verify_pipeline_core(
    BatchVerifyBuffers *buf,
    const unsigned char *d_msgs,
    size_t mlen,
    const unsigned char *d_pre,
    size_t prelen,
    int batch_count,
    int *h_results,
    hipStream_t stream = 0)
{
    if (batch_count <= 0 || batch_count > buf->max_batch) return -1;

    /* 初始化 results 和 hint 缓冲区 */
    hipMemsetAsync(buf->d_results, 0, batch_count * sizeof(int), stream);
    hipMemsetAsync(buf->d_h, 0,
                    (size_t)batch_count * PARAM_K * PARAM_N * sizeof(coeff_t),
                    stream);

    /* [1] 解包签名 → z, h, challenge 数据 */
    {
        int tpb = 64, nblk = (batch_count + tpb - 1) / tpb;
        batch_verify_unpack_kernel<<<nblk, tpb, 0, stream>>>(
            buf->d_z, buf->d_h, buf->d_cbuf, buf->d_results,
            buf->d_raw_sigs, batch_count);
    }

    /* [2] 范数检查 z */
    {
        int tpb = 64, nblk = (batch_count + tpb - 1) / tpb;
        batch_verify_chknorm_z_kernel<<<nblk, tpb, 0, stream>>>(
            buf->d_results, buf->d_z, batch_count);
    }

    /* [3] NTT(z) — shared-memory batch */
    launch_batch_ntt(buf->d_z, batch_count * PARAM_L, stream);

    /* [4] w = A · z_hat (共享矩阵, 2D grid) */
    {
        dim3 grid(batch_count, PARAM_K);
        batch_verify_matvec_kernel<<<grid, PARAM_N, 0, stream>>>(
            buf->d_w, buf->d_mat, buf->d_z, batch_count);
    }

    /* [5] 生成挑战多项式 cp */
#if ALGORITHM == ALGO_AIGIS
    /* Aigis: d_cbuf 已包含完整挑战多项式, 直接复制 */
    hipMemcpyAsync(buf->d_cp, buf->d_cbuf,
               (size_t)batch_count * PARAM_N * sizeof(coeff_t),
               hipMemcpyDeviceToDevice, stream);
#else
    /* ML-DSA: 从 ctilde 种子重建挑战多项式 */
    {
        int tpb = 64, nblk = (batch_count + tpb - 1) / tpb;
        batch_verify_challenge_kernel<<<nblk, tpb, 0, stream>>>(
            buf->d_cp, buf->d_cbuf, batch_count);
    }
#endif

    /* [6] NTT(cp) */
    launch_batch_ntt(buf->d_cp, batch_count, stream);

    /* [7] w -= cp · t1_hat (2D grid) */
    {
        dim3 grid(batch_count, PARAM_K);
        batch_verify_sub_cp_t1_kernel<<<grid, PARAM_N, 0, stream>>>(
            buf->d_w, buf->d_cp, buf->d_t1_hat, batch_count);
    }

    /* [8] reduce + INVNTT + normalize */
    launch_batch_reduce(buf->d_w, batch_count * PARAM_K * PARAM_N, stream);
    launch_batch_invntt(buf->d_w, batch_count * PARAM_K, stream);
#if ALGORITHM == ALGO_AIGIS
    launch_batch_freeze2q(buf->d_w, PARAM_K * batch_count, stream);
#else
    launch_batch_reduce(buf->d_w, batch_count * PARAM_K * PARAM_N, stream);
    launch_batch_caddq(buf->d_w, batch_count * PARAM_K * PARAM_N, stream);
#endif

    /* [9] w1 = use_hint(w, h) */
    launch_batch_use_hint(buf->d_w1, buf->d_w, buf->d_h,
                          batch_count * PARAM_K * PARAM_N, stream);

    /* [10] mu = H(tr || [pre ||] m) */
    {
        int tpb = 32, nblk = (batch_count + tpb - 1) / tpb;
        batch_verify_compute_mu_kernel<<<nblk, tpb, 0, stream>>>(
            buf->d_mu, buf->d_tr, d_msgs, mlen,
            d_pre, prelen,
            batch_count);
    }

    /* [11] 最终比较: H(mu || pack(w1)) vs 原始挑战 */
    {
#if ALGORITHM == ALGO_AIGIS
        int tpb = 32;
#else
        int tpb = 128;
#endif
        int nblk = (batch_count + tpb - 1) / tpb;
        batch_verify_compare_kernel<<<nblk, tpb, 0, stream>>>(
            buf->d_results, buf->d_mu, buf->d_w1, buf->d_cbuf, batch_count);
    }

    /* 回传结果 */
    hipMemcpyAsync(h_results, buf->d_results,
                    batch_count * sizeof(int), hipMemcpyDeviceToHost,
                    stream);
    hipStreamSynchronize(stream);

    return 0;
}

/* ================================================================
 * 批量验证 pipeline (host 签名输入)
 * ================================================================ */
static int batch_verify_pipeline(
    BatchVerifyBuffers *buf,
    const unsigned char *h_sigs,
    const unsigned char *d_msgs,
    size_t mlen,
    const unsigned char *d_pre,
    size_t prelen,
    int batch_count,
    int *h_results,
    hipStream_t stream = 0)
{
    if (batch_count <= 0 || batch_count > buf->max_batch) return -1;

    hipMemcpyAsync(buf->d_raw_sigs, h_sigs,
                    (size_t)batch_count * CRYPTO_BYTES,
                    hipMemcpyHostToDevice,
                    stream);

    return batch_verify_pipeline_core(buf, d_msgs, mlen, d_pre, prelen,
                                      batch_count, h_results, stream);
}

/* ================================================================
 * 批量验证 pipeline (device 签名输入)
 * ================================================================ */
static int batch_verify_pipeline_device_sigs(
    BatchVerifyBuffers *buf,
    const unsigned char *d_sigs,
    const unsigned char *d_msgs,
    size_t mlen,
    const unsigned char *d_pre,
    size_t prelen,
    int batch_count,
    int *h_results,
    hipStream_t stream = 0)
{
    if (batch_count <= 0 || batch_count > buf->max_batch) return -1;

    if (d_sigs != buf->d_raw_sigs) {
        hipMemcpyAsync(buf->d_raw_sigs, d_sigs,
                        (size_t)batch_count * CRYPTO_BYTES,
                        hipMemcpyDeviceToDevice,
                        stream);
    }

    return batch_verify_pipeline_core(buf, d_msgs, mlen, d_pre, prelen,
                                      batch_count, h_results, stream);
}

#endif /* BATCH_VERIFY_CUH */
