#include "hip/hip_runtime.h"
/*
 * batch_sign.cuh — 分解式批量签名 pipeline (算子级并行)
 *
 * 核心优化思路:
 *   1. y 采样 (expand_gamma1) 全批次并行 — 每实例独立 SHAKE 流
 *   2. NTT(y) 使用 shared-memory 批量 kernel (128 线程/poly)
 *   3. w = A·y_hat 使用共享矩阵 2D grid matvec (复用 verify matvec)
 *   4. INVNTT + 归一化 + 分解 + 哈希挑战 — 专用批量 kernel
 *   5. z/cs2/ct0 计算: 挑战多项式 × 共享向量 → 批量 pointwise + INVNTT
 *
 * Rejection loop: 每轮全批次并行; 已完成实例通过 d_done 标志跳过.
 *
 * 所有实例共用:
 *   mat (矩阵 A, NTT域), s1_ntt, s2_ntt, t0_ntt, mu, rhoprime/key_mu
 * Per-instance 差异化:
 *   y — 通过 per-instance 初始 nonce (inst 号) 保证唯一性
 */

#ifndef BATCH_SIGN_CUH
#define BATCH_SIGN_CUH

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

/* 最大拒绝轮数 — ML-DSA-44 每轮接受率≈23%，P(>50轮)≈0.765^50≈1.5e-6/inst
 * B=4096 时期望失败≈0.006，基本为零；Aigis/ML-DSA其他参数类似 */
#define MAX_SIGN_ROUNDS 200

#ifndef BATCH_SIGN_DECOMP_SYNC_EACH_ROUND
#define BATCH_SIGN_DECOMP_SYNC_EACH_ROUND 0
#endif

#ifndef BATCH_SIGN_DECOMP_CHECK_INTERVAL
#define BATCH_SIGN_DECOMP_CHECK_INTERVAL 4
#endif

#ifndef BATCH_SIGN_SAMPLE_TPB
#define BATCH_SIGN_SAMPLE_TPB 64
#endif

#ifndef BATCH_SIGN_HASH_TPB
#define BATCH_SIGN_HASH_TPB 32
#endif

#ifndef BATCH_SIGN_CHECK_TPB
#define BATCH_SIGN_CHECK_TPB 32
#endif

#ifndef BATCH_SIGN_DECOMP_TAIL_ENABLE
#define BATCH_SIGN_DECOMP_TAIL_ENABLE 0
#endif

#ifndef BATCH_SIGN_DECOMP_TAIL_AFTER
#define BATCH_SIGN_DECOMP_TAIL_AFTER 24
#endif

#ifndef BATCH_SIGN_DECOMP_TAIL_PENDING_DIV
#define BATCH_SIGN_DECOMP_TAIL_PENDING_DIV 128
#endif

#ifndef BATCH_SIGN_DECOMP_TAIL_PENDING_MIN
#define BATCH_SIGN_DECOMP_TAIL_PENDING_MIN 16
#endif

#ifndef BATCH_SIGN_SAMPLE_DUP_YHAT
#define BATCH_SIGN_SAMPLE_DUP_YHAT 0
#endif

#ifndef BATCH_SIGN_CP_FUSE_ENABLE
#define BATCH_SIGN_CP_FUSE_ENABLE 0
#endif

/* ================================================================
 * 共享材料 (来自 precomp_t, 一次提取, 所有实例共用)
 * ================================================================ */
struct BatchSignShared {
    coeff_t  *d_mat;        /* K * L * N — 矩阵 A (NTT 域, SoA, 无 batch 维) */
    coeff_t  *d_s1_ntt;     /* L * N */
    coeff_t  *d_s2_ntt;     /* K * N */
    coeff_t  *d_t0_ntt;     /* K * N */
    uint8_t  *d_mu;         /* CRHBYTES */
    uint8_t  *d_rhoprime;   /* ML-DSA: CRHBYTES  /  Aigis: SEEDBYTES+CRHBYTES */
};

/* ================================================================
 * per-instance 工作缓冲区
 * ================================================================ */
struct BatchSignPipeline {
    BatchSignShared sh;

    coeff_t  *d_y;          /* L * B * N SoA — y (系数域, 全程保留) */
    coeff_t  *d_y_hat;      /* L * B * N SoA — NTT(y) (供 matvec 使用, 之后可覆盖) */
    coeff_t  *d_w;          /* K * B * N SoA — A·y 归一化后 (Aigis 保留供 wcs2) */
    coeff_t  *d_w0;         /* K * B * N SoA — decompose 低位 */
    coeff_t  *d_w1;         /* K * B * N SoA — decompose 高位 */
    coeff_t  *d_cp;         /* B * N SoA — 挑战多项式 (NTT 域) */
    coeff_t  *d_z;          /* L * B * N SoA — INVNTT(cp·s1)+y */
    coeff_t  *d_cs2;        /* K * B * N SoA — INVNTT(cp·s2) */
    coeff_t  *d_ct0;        /* K * B * N SoA — INVNTT(cp·t0) */
    uint8_t  *d_cbuf;       /* B*CTILDEBYTES (ML-DSA) or B*N*4 (Aigis cp poly) */
    uint16_t *d_nonces;     /* B — per-instance 当前 nonce */
    int      *d_done;       /* B — 0=pending, 1=done */
    int      *d_done_count; /* B completed signatures */
    uint8_t  *d_sigs;       /* B * CRYPTO_BYTES — 输出签名 (AoS) */

    int       max_batch;
};

struct BatchSignRuntimeOptions {
    int cp_fuse_enable;
    int check_interval;
    int hash_tpb;
    int check_tpb;
};

static BatchSignRuntimeOptions batch_sign_default_runtime_options(void) {
    BatchSignRuntimeOptions opt;
    opt.cp_fuse_enable = BATCH_SIGN_CP_FUSE_ENABLE;
    opt.check_interval = BATCH_SIGN_DECOMP_CHECK_INTERVAL;
    opt.hash_tpb = BATCH_SIGN_HASH_TPB;
    opt.check_tpb = BATCH_SIGN_CHECK_TPB;
    return opt;
}

/* ================================================================
 * [0a] 共享材料提取 — precomp_t AoS → flat SoA
 * ================================================================ */
__global__ void batch_sign_setup_kernel(
    coeff_t        *d_mat_flat,
    coeff_t        *d_s1_flat,
    coeff_t        *d_s2_flat,
    coeff_t        *d_t0_flat,
    const precomp_t *d_pc)
{
    for (int k = 0; k < PARAM_K; k++)
        for (int l = 0; l < PARAM_L; l++)
            for (int c = 0; c < PARAM_N; c++)
                d_mat_flat[(k * PARAM_L + l) * PARAM_N + c]
                    = d_pc->mat[k].vec[l].coeffs[c];
    for (int l = 0; l < PARAM_L; l++)
        for (int c = 0; c < PARAM_N; c++)
            d_s1_flat[l * PARAM_N + c] = d_pc->s1_ntt.vec[l].coeffs[c];
    for (int k = 0; k < PARAM_K; k++)
        for (int c = 0; c < PARAM_N; c++) {
            d_s2_flat[k * PARAM_N + c] = d_pc->s2_ntt.vec[k].coeffs[c];
            d_t0_flat[k * PARAM_N + c] = d_pc->t0_ntt.vec[k].coeffs[c];
        }
}

/* ================================================================
 * [0b] 计算 mu 和 rhoprime (单线程)
 * ================================================================ */
__global__ void batch_sign_compute_mu_rhoprime_kernel(
    uint8_t        *d_mu,
    uint8_t        *d_rhoprime,
    const precomp_t *d_pc,
    const uint8_t  *d_msg,
    size_t          mlen,
    const uint8_t  *d_pre,
    size_t          prelen,
    const uint8_t  *d_rnd)
{
    keccak_state state;
#if ALGORITHM == ALGO_MLDSA
    shake256_init(&state);
    shake256_absorb(&state, d_pc->tr, TRBYTES);
    shake256_absorb(&state, d_pre, prelen);
    shake256_absorb(&state, d_msg, mlen);
    shake256_finalize(&state);
    shake256_squeeze(d_mu, CRHBYTES, &state);

    shake256_init(&state);
    shake256_absorb(&state, d_pc->key, SEEDBYTES);
#if RNDBYTES > 0
    shake256_absorb(&state, d_rnd, RNDBYTES);
#endif
    shake256_absorb(&state, d_mu, CRHBYTES);
    shake256_finalize(&state);
    shake256_squeeze(d_rhoprime, CRHBYTES, &state);

#elif ALGORITHM == ALGO_AIGIS
    shake256_init(&state);
    shake256_absorb(&state, d_pc->tr, TRBYTES);
    shake256_absorb(&state, d_msg, mlen);
    shake256_finalize(&state);
    shake256_squeeze(d_mu, CRHBYTES, &state);
    /* key_mu = key || mu */
    for (int i = 0; i < SEEDBYTES; i++) d_rhoprime[i]           = d_pc->key[i];
    for (int i = 0; i < CRHBYTES;  i++) d_rhoprime[SEEDBYTES+i] = d_mu[i];
#endif
    (void)d_rnd; (void)prelen;
}

/* ================================================================
 * [0c] 初始化 nonce / done
 * ================================================================ */
__global__ void batch_sign_init_kernel(uint16_t *d_nonces, int *d_done, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
#if ALGORITHM == ALGO_MLDSA
    d_nonces[i] = (uint16_t)i;
#else
    d_nonces[i] = (uint16_t)((unsigned)i * PARAM_L);
#endif
    d_done[i] = 0;
}

/* ================================================================
 * [1] 采样 y kernel — 并行 expand_gamma1
 *     每线程: 一个 pending 实例 → L 个多项式
 * ================================================================ */
__global__ void __launch_bounds__(BATCH_SIGN_SAMPLE_TPB)
batch_sign_sample_y_kernel(
    coeff_t        *d_y,
#if BATCH_SIGN_SAMPLE_DUP_YHAT
    coeff_t        *d_y_hat,
#endif
    uint16_t       *d_nonces,
    const int      *d_done,
    const uint8_t  *d_rhoprime,
    int             B)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= B) return;
#if BATCH_SIGN_SAMPLE_DUP_YHAT
    if (d_done[inst]) {
        for (int l = 0; l < PARAM_L; l++) {
            const coeff_t *src = d_y + (size_t)l * B * PARAM_N + (size_t)inst * PARAM_N;
            coeff_t *dst_hat = d_y_hat + (size_t)l * B * PARAM_N + (size_t)inst * PARAM_N;
            for (int c = 0; c < PARAM_N; c++) dst_hat[c] = src[c];
        }
        return;
    }
#else
    if (d_done[inst]) return;
#endif

    uint16_t base = d_nonces[inst];
#if ALGORITHM == ALGO_MLDSA
    d_nonces[inst] = (uint16_t)(base + 1);
#else
    d_nonces[inst] = (uint16_t)(base + (uint16_t)PARAM_L);
#endif
    for (int l = 0; l < PARAM_L; l++) {
        poly tmp;
        poly_uniform_gamma1(&tmp, d_rhoprime, GAMMA1_NONCE(base, l));
        coeff_t *dst = d_y + (size_t)l * B * PARAM_N + (size_t)inst * PARAM_N;
#if BATCH_SIGN_SAMPLE_DUP_YHAT
        coeff_t *dst_hat = d_y_hat + (size_t)l * B * PARAM_N + (size_t)inst * PARAM_N;
        for (int c = 0; c < PARAM_N; c++) {
            coeff_t v = tmp.coeffs[c];
            dst[c] = v;
            dst_hat[c] = v;
        }
#else
        for (int c = 0; c < PARAM_N; c++) dst[c] = tmp.coeffs[c];
#endif
    }
}

/* ================================================================
 * [6] 分解 w → (w1, w0) — per-coeff batch kernel
 * ================================================================ */
__global__ void batch_sign_decompose_kernel(
    coeff_t        *d_w1,
    coeff_t        *d_w0,
    const coeff_t  *d_w_in,
    int             total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int32_t a0;
    int32_t a1 = decompose(&a0, d_w_in[idx]);
    d_w1[idx] = a1;
    d_w0[idx] = a0;
}

/* ================================================================
 * [7] 哈希挑战 — H(mu || pack(w1)) → cp  (并存 c_seed / c_poly 到 d_cbuf)
 * ================================================================ */
__global__ void __launch_bounds__(BATCH_SIGN_HASH_TPB)
batch_sign_hash_cp_kernel(
    coeff_t        *d_cp,
    uint8_t        *d_cbuf,
    const uint8_t  *d_mu,
    const coeff_t  *d_w1,
    const int      *d_done,
    int             B)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= B || d_done[inst]) return;

    /* 打包 w1 */
    uint8_t w1_packed[PARAM_K * POLYW1_PACKEDBYTES];
    for (int ki = 0; ki < PARAM_K; ki++) {
        const coeff_t *w1k = d_w1 + (size_t)ki * B * PARAM_N + (size_t)inst * PARAM_N;
        uint8_t *r = w1_packed + ki * POLYW1_PACKEDBYTES;
#if ALGORITHM == ALGO_MLDSA
  #if PARAM_GAMMA2 == ((PARAM_Q-1)/88)
        for (unsigned i = 0; i < PARAM_N/4; i++) {
            r[3*i+0]  = (uint8_t)(w1k[4*i+0]);
            r[3*i+0] |= (uint8_t)(w1k[4*i+1] << 6);
            r[3*i+1]  = (uint8_t)(w1k[4*i+1] >> 2);
            r[3*i+1] |= (uint8_t)(w1k[4*i+2] << 4);
            r[3*i+2]  = (uint8_t)(w1k[4*i+2] >> 4);
            r[3*i+2] |= (uint8_t)(w1k[4*i+3] << 2);
        }
  #else /* GAMMA2 = (Q-1)/32 */
        for (unsigned i = 0; i < PARAM_N/2; i++)
            r[i] = (uint8_t)(w1k[2*i+0] | (w1k[2*i+1] << 4));
  #endif
#elif ALGORITHM == ALGO_AIGIS
        for (unsigned i = 0; i < PARAM_N/8; i++) {
            r[3*i+0] = (uint8_t)(w1k[8*i+0] | (w1k[8*i+1] << 3) | (w1k[8*i+2] << 6));
            r[3*i+1] = (uint8_t)((w1k[8*i+2] >> 2) | (w1k[8*i+3] << 1)
                      | (w1k[8*i+4] << 4) | (w1k[8*i+5] << 7));
            r[3*i+2] = (uint8_t)((w1k[8*i+5] >> 1) | (w1k[8*i+6] << 2) | (w1k[8*i+7] << 5));
        }
#endif
    }

    coeff_t *cp = d_cp + (size_t)inst * PARAM_N;

#if ALGORITHM == ALGO_MLDSA
    {
        keccak_state st;
        shake256_init(&st);
        shake256_absorb(&st, d_mu, CRHBYTES);
        shake256_absorb(&st, w1_packed, PARAM_K * POLYW1_PACKEDBYTES);
        shake256_finalize(&st);
        uint8_t c_seed[CTILDEBYTES];
        shake256_squeeze(c_seed, CTILDEBYTES, &st);
        /* 保存 c_seed */
        uint8_t *cbuf = d_cbuf + (size_t)inst * CTILDEBYTES;
        for (int i = 0; i < CTILDEBYTES; i++) cbuf[i] = c_seed[i];
        /* SampleInBall → cp (与 batch_verify_challenge_kernel 逻辑相同) */
        uint8_t buf2[SHAKE256_RATE];
        keccak_state st2;
        shake256_init(&st2);
        shake256_absorb(&st2, c_seed, CTILDEBYTES);
        shake256_finalize(&st2);
        shake256_squeezeblocks(buf2, 1, &st2);
        uint64_t signs = 0;
        for (int i = 0; i < 8; i++) signs |= (uint64_t)buf2[i] << (8*i);
        unsigned int pos = 8;
        for (int i = 0; i < PARAM_N; i++) cp[i] = 0;
        for (int i = PARAM_N - PARAM_TAU; i < PARAM_N; i++) {
            unsigned int b;
            do {
                if (pos >= SHAKE256_RATE) { shake256_squeezeblocks(buf2, 1, &st2); pos = 0; }
                b = buf2[pos++];
            } while (b > i);
            cp[i] = cp[b];
            cp[b] = 1 - 2*(int)(signs & 1);
            signs >>= 1;
        }
    }
#elif ALGORITHM == ALGO_AIGIS
    {
        poly c_tmp;
        poly_challenge(&c_tmp, d_mu, w1_packed, PARAM_K * POLYW1_PACKEDBYTES);
        for (int i = 0; i < PARAM_N; i++) cp[i] = c_tmp.coeffs[i];
        /* 保存 cp 供打包签名 */
        coeff_t *cbuf_cp = (coeff_t *)(d_cbuf + (size_t)inst * PARAM_N * sizeof(coeff_t));
        for (int i = 0; i < PARAM_N; i++) cbuf_cp[i] = c_tmp.coeffs[i];
    }
#endif
}

/* ================================================================
 * [9a] cp_ntt × shared_vec → out  (2D grid: (B, poly_count) × N threads)
 *      out[poly][inst][tid] = mont(cp[inst][tid] * shared[poly][tid])
 * ================================================================ */
__global__ void batch_sign_pointwise_cp_shared_kernel(
    coeff_t        *d_out,
    const coeff_t  *d_cp,
    const coeff_t  *d_shared,
    int             poly_count,
    int             B)
{
    int inst = blockIdx.x;
    int poly = blockIdx.y;
    int tid  = threadIdx.x;
    if (inst >= B || poly >= poly_count) return;
    coeff_t c = d_cp[(size_t)inst * PARAM_N + tid];
    coeff_t s = d_shared[(size_t)poly * PARAM_N + tid];
    d_out[(size_t)poly * B * PARAM_N + (size_t)inst * PARAM_N + tid]
        = (coeff_t)montgomery_reduce((coeff2_t)c * s);
}

__global__ void __launch_bounds__(256)
batch_sign_pointwise_cp_all_shared_kernel(
    coeff_t        *d_z,
    coeff_t        *d_cs2,
    coeff_t        *d_ct0,
    const coeff_t  *d_cp,
    const coeff_t  *d_s1_ntt,
    const coeff_t  *d_s2_ntt,
    const coeff_t  *d_t0_ntt,
    int             B)
{
    int inst = blockIdx.x;
    int poly = blockIdx.y;
    int tid  = threadIdx.x;
    if (inst >= B || tid >= PARAM_N) return;

    coeff_t c = d_cp[(size_t)inst * PARAM_N + tid];
    size_t out_idx = (size_t)poly * B * PARAM_N + (size_t)inst * PARAM_N + tid;

    if (poly < PARAM_L) {
        coeff_t s1 = d_s1_ntt[(size_t)poly * PARAM_N + tid];
        d_z[out_idx] = (coeff_t)montgomery_reduce((coeff2_t)c * s1);
    }
    if (poly < PARAM_K) {
        coeff_t s2 = d_s2_ntt[(size_t)poly * PARAM_N + tid];
        coeff_t t0 = d_t0_ntt[(size_t)poly * PARAM_N + tid];
        d_cs2[out_idx] = (coeff_t)montgomery_reduce((coeff2_t)c * s2);
        d_ct0[out_idx] = (coeff_t)montgomery_reduce((coeff2_t)c * t0);
    }
}

/* ================================================================
 * [9b] z += y  (in-place add: d_z[i] += d_y[i])
 * ================================================================ */
__global__ void batch_sign_add_y_kernel(
    coeff_t        *d_z,
    const coeff_t  *d_y,
    int             total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) d_z[idx] += d_y[idx];
}

/* ================================================================
 * [12] 范数检查 + 提示计算 + 打包签名 (per-instance, 单线程)
 * ================================================================ */
__global__ void __launch_bounds__(BATCH_SIGN_CHECK_TPB)
batch_sign_check_pack_kernel(
    int            *d_done,
    uint8_t        *d_sigs,
    const coeff_t  *d_z,
    const coeff_t  *d_w,     /* 归一化 w (供 Aigis wcs2) */
    const coeff_t  *d_w0,    /* decompose 低位 */
    const coeff_t  *d_w1,    /* decompose 高位 */
    const coeff_t  *d_cs2,
    const coeff_t  *d_ct0,
    const uint8_t  *d_cbuf,
    int             B)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= B || d_done[inst]) return;

    /* 加载 z */
    polyvecl z_loc;
    for (int l = 0; l < PARAM_L; l++)
        for (int c = 0; c < PARAM_N; c++)
            z_loc.vec[l].coeffs[c] =
                d_z[(size_t)l * B * PARAM_N + (size_t)inst * PARAM_N + c];
#if ALGORITHM == ALGO_MLDSA
    polyvecl_reduce(&z_loc);
    if (polyvecl_chknorm(&z_loc, PARAM_GAMMA1 - PARAM_BETA1)) return;
#else
    polyvecl_freeze4q(&z_loc);
    if (polyvecl_chknorm(&z_loc, PARAM_GAMMA1 - PARAM_BETA1)) return;
#endif

#if ALGORITHM == ALGO_MLDSA
    /* r0 = w0 - cs2 */
    polyveck r0, ct0_loc, w1_loc, h_loc;
    for (int k = 0; k < PARAM_K; k++)
        for (int c = 0; c < PARAM_N; c++) {
            int32_t w0v  = d_w0 [(size_t)k * B * PARAM_N + (size_t)inst * PARAM_N + c];
            int32_t cs2v = d_cs2[(size_t)k * B * PARAM_N + (size_t)inst * PARAM_N + c];
            r0.vec[k].coeffs[c] = coeff_sub(w0v, cs2v);
        }
    polyveck_reduce(&r0);
    if (polyveck_chknorm(&r0, PARAM_GAMMA2 - PARAM_BETA2)) return;

    for (int k = 0; k < PARAM_K; k++)
        for (int c = 0; c < PARAM_N; c++)
            ct0_loc.vec[k].coeffs[c] =
                d_ct0[(size_t)k * B * PARAM_N + (size_t)inst * PARAM_N + c];
    polyveck_reduce(&ct0_loc);
    if (polyveck_chknorm(&ct0_loc, PARAM_GAMMA2)) return;

    for (int k = 0; k < PARAM_K; k++)
        for (int c = 0; c < PARAM_N; c++)
            w1_loc.vec[k].coeffs[c] =
                d_w1[(size_t)k * B * PARAM_N + (size_t)inst * PARAM_N + c];

    polyveck_add(&r0, &r0, &ct0_loc);           /* w0_adj = r0 + ct0 */
    unsigned int n = polyveck_make_hint(&h_loc, &r0, &w1_loc);
    if (n > PARAM_OMEGA) return;

    uint8_t *sig_out = d_sigs + (size_t)inst * CRYPTO_BYTES;
    const uint8_t *c_seed = d_cbuf + (size_t)inst * CTILDEBYTES;
    pack_sig(sig_out, c_seed, &z_loc, &h_loc);

#elif ALGORITHM == ALGO_AIGIS
    /* wcs2 = w - cs2 */
    polyveck wcs2, ct0_loc, h_loc;
    for (int k = 0; k < PARAM_K; k++)
        for (int c = 0; c < PARAM_N; c++) {
            int32_t wv   = d_w  [(size_t)k * B * PARAM_N + (size_t)inst * PARAM_N + c];
            int32_t cs2v = d_cs2[(size_t)k * B * PARAM_N + (size_t)inst * PARAM_N + c];
            wcs2.vec[k].coeffs[c] = coeff_sub(wv, cs2v);
        }
    polyveck_freeze4q(&wcs2);

    /* w1 consistency: decompose(wcs2)[high] == d_w1 */
    {
        polyveck wcs2_high, wcs2_low;
        polyveck_decompose(&wcs2_high, &wcs2_low, &wcs2);
        polyveck_freeze2q(&wcs2_low);
        for (int k = 0; k < PARAM_K; k++)
            for (int c = 0; c < PARAM_N; c++) {
                int32_t w1v = d_w1[(size_t)k * B * PARAM_N + (size_t)inst * PARAM_N + c];
                if (wcs2_high.vec[k].coeffs[c] != w1v) return;
            }
        if (polyveck_chknorm(&wcs2_low, PARAM_GAMMA2 - PARAM_BETA2)) return;
    }

    for (int k = 0; k < PARAM_K; k++)
        for (int c = 0; c < PARAM_N; c++)
            ct0_loc.vec[k].coeffs[c] =
                d_ct0[(size_t)k * B * PARAM_N + (size_t)inst * PARAM_N + c];
    polyveck_freeze2q(&ct0_loc);
    if (polyveck_chknorm(&ct0_loc, PARAM_GAMMA2)) return;

    /* make_hint(wcs2+ct0, -ct0) */
    polyveck tmp_loc, neg_ct0;
    polyveck_add(&tmp_loc, &wcs2, &ct0_loc);
    neg_ct0 = ct0_loc;
    polyveck_neg(&neg_ct0);
    polyveck_freeze2q(&tmp_loc);
    unsigned int n = polyveck_make_hint(&h_loc, &tmp_loc, &neg_ct0);
    if (n > PARAM_OMEGA) return;

    poly c_poly;
    const coeff_t *cbuf_cp = (const coeff_t *)(d_cbuf + (size_t)inst * PARAM_N * sizeof(coeff_t));
    for (int i = 0; i < PARAM_N; i++) c_poly.coeffs[i] = cbuf_cp[i];
    uint8_t *sig_out = d_sigs + (size_t)inst * CRYPTO_BYTES;
    pack_sig(sig_out, &z_loc, &h_loc, &c_poly);
#endif

    d_done[inst] = 1;
}

__global__ void batch_sign_count_done_kernel(const int *d_done, int *d_done_count, int B) {
    __shared__ int local_count;
    if (threadIdx.x == 0) local_count = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B && d_done[idx]) atomicAdd(&local_count, 1);
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(d_done_count, local_count);
}

/* Finish the small rejection tail without launching full-batch NTT/matvec rounds. */
__global__ void __launch_bounds__(64, 1)
batch_sign_tail_precomp_kernel(
    int             *d_done,
    uint8_t         *d_sigs,
    const uint16_t  *d_nonces,
    const precomp_t *d_pc,
    const uint8_t   *d_mu,
    const uint8_t   *d_rhoprime,
    int              B)
{
    int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= B || d_done[inst]) return;

    size_t siglen = 0;
#if ALGORITHM == ALGO_MLDSA
    int r = crypto_sign_signature_precomp_cached(
        d_sigs + (size_t)inst * CRYPTO_BYTES, &siglen,
        d_mu, d_rhoprime, d_pc, d_nonces[inst]);
#else
    int r = crypto_sign_signature_precomp_cached(
        d_sigs + (size_t)inst * CRYPTO_BYTES, &siglen,
        d_mu, d_rhoprime, d_pc, d_nonces[inst]);
#endif
    if (r == 0 && siglen == CRYPTO_BYTES) d_done[inst] = 1;
}

/* ================================================================
 * Host API — 缓冲区分配/释放
 * ================================================================ */
static int batch_sign_alloc(BatchSignPipeline *p, int max_batch) {
    memset(p, 0, sizeof(*p));
    p->max_batch = max_batch;
    size_t B = max_batch, N = PARAM_N;

#define BS_TRY(ptr, sz) do { \
    if (hipMalloc(&(ptr), (sz)) != hipSuccess) { hipGetLastError(); return -1; } \
} while(0)

    BS_TRY(p->sh.d_mat,        (size_t)PARAM_K * PARAM_L * N * sizeof(coeff_t));
    BS_TRY(p->sh.d_s1_ntt,     (size_t)PARAM_L * N * sizeof(coeff_t));
    BS_TRY(p->sh.d_s2_ntt,     (size_t)PARAM_K * N * sizeof(coeff_t));
    BS_TRY(p->sh.d_t0_ntt,     (size_t)PARAM_K * N * sizeof(coeff_t));
    BS_TRY(p->sh.d_mu,         CRHBYTES);
#if ALGORITHM == ALGO_MLDSA
    BS_TRY(p->sh.d_rhoprime,   CRHBYTES);
#else
    BS_TRY(p->sh.d_rhoprime,   SEEDBYTES + CRHBYTES);
#endif
    BS_TRY(p->d_y,     (size_t)PARAM_L * B * N * sizeof(coeff_t));
    BS_TRY(p->d_y_hat, (size_t)PARAM_L * B * N * sizeof(coeff_t));
    BS_TRY(p->d_w,     (size_t)PARAM_K * B * N * sizeof(coeff_t));
    BS_TRY(p->d_w0,    (size_t)PARAM_K * B * N * sizeof(coeff_t));
    BS_TRY(p->d_w1,    (size_t)PARAM_K * B * N * sizeof(coeff_t));
    BS_TRY(p->d_cp,    (size_t)B * N * sizeof(coeff_t));
    BS_TRY(p->d_z,     (size_t)PARAM_L * B * N * sizeof(coeff_t));
    BS_TRY(p->d_cs2,   (size_t)PARAM_K * B * N * sizeof(coeff_t));
    BS_TRY(p->d_ct0,   (size_t)PARAM_K * B * N * sizeof(coeff_t));
    BS_TRY(p->d_nonces, B * sizeof(uint16_t));
    BS_TRY(p->d_done,   B * sizeof(int));
    BS_TRY(p->d_done_count, sizeof(int));
    BS_TRY(p->d_sigs,   B * CRYPTO_BYTES);
#if ALGORITHM == ALGO_MLDSA
    BS_TRY(p->d_cbuf, B * CTILDEBYTES);
#else
    BS_TRY(p->d_cbuf, B * N * sizeof(coeff_t));
#endif
#undef BS_TRY
    return 0;
}

static void batch_sign_free(BatchSignPipeline *p) {
    hipFree(p->sh.d_mat);     hipFree(p->sh.d_s1_ntt);
    hipFree(p->sh.d_s2_ntt);  hipFree(p->sh.d_t0_ntt);
    hipFree(p->sh.d_mu);      hipFree(p->sh.d_rhoprime);
    hipFree(p->d_y);    hipFree(p->d_y_hat); hipFree(p->d_w);
    hipFree(p->d_w0);   hipFree(p->d_w1);    hipFree(p->d_cp);
    hipFree(p->d_z);    hipFree(p->d_cs2);   hipFree(p->d_ct0);
    hipFree(p->d_nonces); hipFree(p->d_done); hipFree(p->d_done_count);
    hipFree(p->d_sigs); hipFree(p->d_cbuf);
    memset(p, 0, sizeof(*p));
}

static int batch_sign_count_done_host(BatchSignPipeline *p, int B) {
    int done_now = 0;
    int tpb = 256, nblk = (B + tpb - 1) / tpb;
    hipMemsetAsync(p->d_done_count, 0, sizeof(int));
    batch_sign_count_done_kernel<<<nblk, tpb>>>(p->d_done, p->d_done_count, B);
    hipMemcpy(&done_now, p->d_done_count, sizeof(int), hipMemcpyDeviceToHost);
    return done_now;
}

static int batch_sign_tail_finish(
    BatchSignPipeline *p,
    int                B,
    const precomp_t   *d_pc)
{
    int tpb = 64, nblk = (B + tpb - 1) / tpb;
    batch_sign_tail_precomp_kernel<<<nblk, tpb>>>(
        p->d_done, p->d_sigs, p->d_nonces, d_pc,
        p->sh.d_mu, p->sh.d_rhoprime, B);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -1;
    return batch_sign_count_done_host(p, B);
}

/* ================================================================
 * 批量签名 pipeline 主函数
 *
 * 前置条件:
 *   p->max_batch >= batch_count
 *   d_pc 已通过 kernel_create_precomp 构造 (device 端)
 *   d_msg / d_rnd / d_pre 均在 device 端
 *
 * 输出:
 *   p->d_sigs[0..batch_count-1*CRYPTO_BYTES] — AoS 格式签名
 * ================================================================ */
static int batch_sign_pipeline_ex(
    BatchSignPipeline *p,
    int                batch_count,
    const precomp_t   *d_pc,
    const uint8_t     *d_msg,
    size_t             mlen,
    const uint8_t     *d_pre,
    size_t             prelen,
    const uint8_t     *d_rnd,
    const BatchSignRuntimeOptions *runtime_opt,
    int               *h_rounds,
    int               *h_done)
{
    if (batch_count <= 0 || batch_count > p->max_batch) return -1;
    int B = batch_count, N = PARAM_N;
    BatchSignRuntimeOptions opt = runtime_opt
        ? *runtime_opt
        : batch_sign_default_runtime_options();
    int runtime_cp_fuse = opt.cp_fuse_enable != 0;
    int runtime_check_interval = opt.check_interval > 0
        ? opt.check_interval
        : BATCH_SIGN_DECOMP_CHECK_INTERVAL;
    int runtime_hash_tpb = opt.hash_tpb > 0 ? opt.hash_tpb : BATCH_SIGN_HASH_TPB;
    int runtime_check_tpb = opt.check_tpb > 0 ? opt.check_tpb : BATCH_SIGN_CHECK_TPB;
    if (runtime_hash_tpb > BATCH_SIGN_HASH_TPB) runtime_hash_tpb = BATCH_SIGN_HASH_TPB;
    if (runtime_check_tpb > BATCH_SIGN_CHECK_TPB) runtime_check_tpb = BATCH_SIGN_CHECK_TPB;

    /* [0a] 提取共享材料 */
    batch_sign_setup_kernel<<<1, 1>>>(
        p->sh.d_mat, p->sh.d_s1_ntt, p->sh.d_s2_ntt, p->sh.d_t0_ntt, d_pc);

    /* [0b] 计算 mu + rhoprime */
    batch_sign_compute_mu_rhoprime_kernel<<<1, 1>>>(
        p->sh.d_mu, p->sh.d_rhoprime, d_pc, d_msg, mlen, d_pre, prelen, d_rnd);

    /* [0c] 初始化 nonce / done */
    {
        int tpb = 256, nblk = (B + tpb - 1) / tpb;
        batch_sign_init_kernel<<<nblk, tpb>>>(p->d_nonces, p->d_done, B);
    }
    /* ============================================================
     * Rejection loop — 固定 MAX_SIGN_ROUNDS 轮, 不在轮间同步.
     * 同一 CUDA stream 内核按顺序执行, 数据依赖自动满足.
     * 已完成实例通过 d_done 标志在各 kernel 内 early-exit.
     * P(B=4096 实例 round>15 未完成) < 1e-10 for ML-DSA-44.
     * ============================================================ */
    for (int round = 0; round < MAX_SIGN_ROUNDS; round++) {

        /* [1] 采样 y */
        {
            int tpb = BATCH_SIGN_SAMPLE_TPB, nblk = (B + tpb - 1) / tpb;
            batch_sign_sample_y_kernel<<<nblk, tpb>>>(
                p->d_y,
#if BATCH_SIGN_SAMPLE_DUP_YHAT
                p->d_y_hat,
#endif
                p->d_nonces, p->d_done, p->sh.d_rhoprime, B);
        }

        /* 备份 y → d_y_hat (后续 NTT 就地覆盖 d_y_hat, 保留 d_y 为系数域) */
#if !BATCH_SIGN_SAMPLE_DUP_YHAT
        hipMemcpyAsync(p->d_y_hat, p->d_y,
                       (size_t)PARAM_L * B * N * sizeof(coeff_t),
                       hipMemcpyDeviceToDevice);
#endif

        /* [2] NTT(y_hat) 就地 */
        launch_batch_ntt(p->d_y_hat, B * PARAM_L);

        /* [3] w = A · y_hat (共享矩阵 2D grid matvec, 复用 verify kernel) */
        {
            dim3 grid(B, PARAM_K);
            batch_verify_matvec_kernel<<<grid, N>>>(
                p->d_w, p->sh.d_mat, p->d_y_hat, B);
        }

        /* [4] reduce + INVNTT(w) */
        launch_batch_reduce(p->d_w, B * PARAM_K * N);
        launch_batch_invntt(p->d_w, B * PARAM_K);

        /* [5] 归一化 w */
#if ALGORITHM == ALGO_MLDSA
        launch_batch_reduce(p->d_w, B * PARAM_K * N);
        launch_batch_caddq(p->d_w, B * PARAM_K * N);
#else
        launch_batch_freeze2q(p->d_w, PARAM_K * B);
#endif

        /* [6] decompose(w) → (w1, w0)
         *     d_w 保留不变 (供 Aigis wcs2 = w - cs2 计算) */
        {
            int total = PARAM_K * B * N;
            int tpb = BATCH_TPB, nblk = (total + tpb - 1) / tpb;
            batch_sign_decompose_kernel<<<nblk, tpb>>>(p->d_w1, p->d_w0, p->d_w, total);
        }

        /* [7] 哈希挑战: H(mu || pack(w1)) → d_cp + d_cbuf */
        {
            int tpb = runtime_hash_tpb, nblk = (B + tpb - 1) / tpb;
            batch_sign_hash_cp_kernel<<<nblk, tpb>>>(
                p->d_cp, p->d_cbuf, p->sh.d_mu, p->d_w1, p->d_done, B);
        }

        /* [8] NTT(cp) 就地 */
        launch_batch_ntt(p->d_cp, B);

        if (runtime_cp_fuse) {
            const int max_shared = (PARAM_L > PARAM_K) ? PARAM_L : PARAM_K;
            dim3 grid_all(B, max_shared);
            batch_sign_pointwise_cp_all_shared_kernel<<<grid_all, N>>>(
                p->d_z, p->d_cs2, p->d_ct0, p->d_cp,
                p->sh.d_s1_ntt, p->sh.d_s2_ntt, p->sh.d_t0_ntt, B);
        }

        /* [9] z = INVNTT(cp_ntt · s1_ntt) + y */
        if (!runtime_cp_fuse) {
            dim3 grid_l(B, PARAM_L);
            batch_sign_pointwise_cp_shared_kernel<<<grid_l, N>>>(
                p->d_z, p->d_cp, p->sh.d_s1_ntt, PARAM_L, B);
        }
        launch_batch_invntt(p->d_z, B * PARAM_L);
        {
            int total = PARAM_L * B * N;
            int tpb = BATCH_TPB, nblk = (total + tpb - 1) / tpb;
            batch_sign_add_y_kernel<<<nblk, tpb>>>(p->d_z, p->d_y, total);
        }

        /* [10] cs2 = INVNTT(cp_ntt · s2_ntt) */
        if (!runtime_cp_fuse) {
            dim3 grid_k(B, PARAM_K);
            batch_sign_pointwise_cp_shared_kernel<<<grid_k, N>>>(
                p->d_cs2, p->d_cp, p->sh.d_s2_ntt, PARAM_K, B);
        }
        launch_batch_invntt(p->d_cs2, B * PARAM_K);

        /* [11] ct0 = INVNTT(cp_ntt · t0_ntt) */
        if (!runtime_cp_fuse) {
            dim3 grid_k(B, PARAM_K);
            batch_sign_pointwise_cp_shared_kernel<<<grid_k, N>>>(
                p->d_ct0, p->d_cp, p->sh.d_t0_ntt, PARAM_K, B);
        }
        launch_batch_invntt(p->d_ct0, B * PARAM_K);

        /* [12] 范数检查 + 提示 + 打包签名 */
        {
            int tpb = runtime_check_tpb, nblk = (B + tpb - 1) / tpb;
            batch_sign_check_pack_kernel<<<nblk, tpb>>>(
                p->d_done, p->d_sigs,
                p->d_z, p->d_w, p->d_w0, p->d_w1,
                p->d_cs2, p->d_ct0, p->d_cbuf, B);
        }
        if (BATCH_SIGN_DECOMP_SYNC_EACH_ROUND ||
            ((round + 1) % runtime_check_interval) == 0 ||
            round + 1 == MAX_SIGN_ROUNDS) {
            int done_now = batch_sign_count_done_host(p, B);
            if (done_now >= B) {
                if (h_rounds) *h_rounds = round + 1;
                if (h_done) *h_done = done_now;
                return 0;
            }
#if BATCH_SIGN_DECOMP_TAIL_ENABLE
            int remaining = B - done_now;
            int tail_limit = B / BATCH_SIGN_DECOMP_TAIL_PENDING_DIV;
            if (tail_limit < BATCH_SIGN_DECOMP_TAIL_PENDING_MIN)
                tail_limit = BATCH_SIGN_DECOMP_TAIL_PENDING_MIN;
            if ((round + 1) >= BATCH_SIGN_DECOMP_TAIL_AFTER &&
                remaining <= tail_limit) {
                int tail_done = batch_sign_tail_finish(p, B, d_pc);
                if (tail_done < 0) return -1;
                if (h_rounds) *h_rounds = round + 1;
                if (h_done) *h_done = tail_done;
                return 0;
            }
#endif
        }
    }

    /* 单次最终同步 — 等待所有轮次完成 */
    hipDeviceSynchronize();
    if (h_rounds) *h_rounds = MAX_SIGN_ROUNDS;
#if BATCH_SIGN_DECOMP_TAIL_ENABLE
    {
        int done_now = batch_sign_count_done_host(p, B);
        if (done_now < B)
            done_now = batch_sign_tail_finish(p, B, d_pc);
        if (h_done) *h_done = done_now;
        if (done_now < 0) return -1;
    }
#else
    if (h_done) *h_done = batch_sign_count_done_host(p, B);
#endif
    return 0;
}

static int batch_sign_pipeline(
    BatchSignPipeline *p,
    int                batch_count,
    const precomp_t   *d_pc,
    const uint8_t     *d_msg,
    size_t             mlen,
    const uint8_t     *d_pre,
    size_t             prelen,
    const uint8_t     *d_rnd)
{
    return batch_sign_pipeline_ex(p, batch_count, d_pc, d_msg, mlen,
                                  d_pre, prelen, d_rnd, nullptr, nullptr, nullptr);
}

#endif /* BATCH_SIGN_CUH */
