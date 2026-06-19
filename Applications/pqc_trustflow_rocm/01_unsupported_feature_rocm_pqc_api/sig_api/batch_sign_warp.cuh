#include "hip/hip_runtime.h"
#ifndef BATCH_SIGN_WARP_CUH
#define BATCH_SIGN_WARP_CUH

#include <stdint.h>
#include <stddef.h>
#include <hip/hip_runtime.h>

#include "params.h"
#include "sign.cuh"
#include "batch_ntt.cuh"

#ifndef BATCH_SIGN_WARP_ENABLE
#define BATCH_SIGN_WARP_ENABLE 1
#endif

#ifndef BATCH_SIGN_WARP_PROFILE
#define BATCH_SIGN_WARP_PROFILE 0
#endif

#define WP_SIGN_WARP_SIZE   32

#define WP_SIGN_WARPS_BLOCK 4


#define WP_SIGN_TPB         (WP_SIGN_WARP_SIZE * WP_SIGN_WARPS_BLOCK)

#if ALGORITHM == ALGO_MLDSA
#define WP_SIGN_SEED_BYTES CRHBYTES
#else
#define WP_SIGN_SEED_BYTES (SEEDBYTES + CRHBYTES)
#endif

#if ALGORITHM == ALGO_AIGIS
#define WP_SIGN_GAMMA1_BUF_BYTES (STREAM256_BLOCKBYTES + 4)
#else
#define WP_SIGN_GAMMA1_BUF_BYTES (POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES)
#endif

enum {
    WP_SIGN_STAT_ATTEMPTS = 0,
    WP_SIGN_STAT_REJ_S2   = 1,
    WP_SIGN_STAT_REJ_Z    = 2,
    WP_SIGN_STAT_REJ_T0   = 3,
    WP_SIGN_STAT_REJ_HINT = 4,
    WP_SIGN_STAT_OK       = 5,
    WP_SIGN_STAT_COUNT    = 6
};

typedef struct {
#if ALGORITHM == ALGO_MLDSA
    uint8_t mu[CRHBYTES];
    uint8_t rhoprime[CRHBYTES];
#else
    uint8_t mu[CRHBYTES];
    uint8_t key_mu[SEEDBYTES + CRHBYTES];
#endif
} wp_sign_cache_t;

typedef struct {
    coeff_t *y;
    coeff_t *w;
    coeff_t *cp;
    coeff_t *tmp;
    uint8_t *packed_w1;
    uint8_t *mu;
    uint8_t *seed;
    uint8_t *work;
} wp_sign_smem_t;

static __host__ __device__ __forceinline__ size_t wp_sign_align(size_t x, size_t a) {
    return (x + a - 1u) & ~(a - 1u);
}

static __host__ __device__ __forceinline__ size_t wp_sign_shared_bytes_per_warp(void) {
    size_t off = 0;
    off = wp_sign_align(off, 16);
    off += (size_t)PARAM_L * PARAM_N * sizeof(coeff_t);
    off = wp_sign_align(off, 16);
    off += (size_t)PARAM_K * PARAM_N * sizeof(coeff_t);
    off = wp_sign_align(off, 16);
    off += (size_t)PARAM_N * sizeof(coeff_t);
    off = wp_sign_align(off, 16);
    off += (size_t)PARAM_N * sizeof(coeff_t);
    off += (size_t)PARAM_K * POLYW1_PACKEDBYTES;
    off = wp_sign_align(off, 16);
    off += CRHBYTES;
    off = wp_sign_align(off, 16);
    off += WP_SIGN_SEED_BYTES;
    off = wp_sign_align(off, 16);
    off += WP_SIGN_GAMMA1_BUF_BYTES;
    return wp_sign_align(off, 16);
}

static inline size_t batch_sign_warp_smem_bytes(void) {
    return wp_sign_shared_bytes_per_warp() * WP_SIGN_WARPS_BLOCK;
}

static inline hipError_t batch_sign_warp_set_smem_attributes(void);

static __device__ __forceinline__ void wp_sign_smem_init(
    wp_sign_smem_t *s, unsigned char *base, int warp_slot)
{
    size_t off = (size_t)warp_slot * wp_sign_shared_bytes_per_warp();
    off = wp_sign_align(off, 16);
    s->y = (coeff_t *)(base + off);
    off += (size_t)PARAM_L * PARAM_N * sizeof(coeff_t);
    off = wp_sign_align(off, 16);
    s->w = (coeff_t *)(base + off);
    off += (size_t)PARAM_K * PARAM_N * sizeof(coeff_t);
    off = wp_sign_align(off, 16);
    s->cp = (coeff_t *)(base + off);
    off += (size_t)PARAM_N * sizeof(coeff_t);
    off = wp_sign_align(off, 16);
    s->tmp = (coeff_t *)(base + off);
    off += (size_t)PARAM_N * sizeof(coeff_t);
    off = wp_sign_align(off, 16);
    s->packed_w1 = base + off;
    off += (size_t)PARAM_K * POLYW1_PACKEDBYTES;
    off = wp_sign_align(off, 16);
    s->mu = base + off;
    off += CRHBYTES;
    off = wp_sign_align(off, 16);
    s->seed = base + off;
    off += WP_SIGN_SEED_BYTES;
    off = wp_sign_align(off, 16);
    s->work = base + off;
}

static __device__ __forceinline__ void wp_sign_store_sig(
    uint8_t *sig_soa, int inst, int N, unsigned int off, uint8_t v)
{
    sig_soa[(size_t)off * (size_t)N + (size_t)inst] = v;
}

static __device__ __forceinline__ int wp_sign_any(int pred) {
    return __ballot_sync(0xffffffffull, pred) != 0u;
}

static __device__ __forceinline__ int wp_sign_coeff_chknorm(coeff_t a, int32_t B) {
#if ALGORITHM == ALGO_MLDSA
    if (B > (PARAM_Q - 1) / 8) return 1;
    int32_t t = a >> 31;
    t = a - (t & 2 * a);
    return t >= B;
#else
    int32_t t = (PARAM_Q - 1) / 2 - a;
    t ^= (t >> 31);
    t = (PARAM_Q - 1) / 2 - t;
    return t >= B;
#endif
}

static __device__ __forceinline__ int wp_sign_poly_chknorm(
    const coeff_t *a, int32_t B, int lane)
{
    int bad = 0;
    for (int i = lane; i < PARAM_N; i += WP_SIGN_WARP_SIZE)
        bad |= wp_sign_coeff_chknorm(a[i], B);
    return wp_sign_any(bad);
}

static __device__ __noinline__ void wp_sign_sample_y_poly(
    coeff_t *dst, const uint8_t *seed, uint16_t nonce, int lane, uint8_t *buf)
{
#if ALGORITHM == ALGO_MLDSA
    if (lane == 0) {
        stream256_state state;
        stream256_init(&state, seed, nonce);
        stream256_squeezeblocks(buf, POLY_UNIFORM_GAMMA1_NBLOCKS, &state);
    }
    __syncwarp();
#if PARAM_GAMMA1 == (1 << 17)
    for (int i = lane; i < PARAM_N / 4; i += WP_SIGN_WARP_SIZE) {
        uint32_t t0 = ((uint32_t)buf[9 * i + 0] |
                      ((uint32_t)buf[9 * i + 1] << 8) |
                      ((uint32_t)buf[9 * i + 2] << 16)) & 0x3ffffu;
        uint32_t t1 = (((uint32_t)buf[9 * i + 2] >> 2) |
                      ((uint32_t)buf[9 * i + 3] << 6) |
                      ((uint32_t)buf[9 * i + 4] << 14)) & 0x3ffffu;
        uint32_t t2 = (((uint32_t)buf[9 * i + 4] >> 4) |
                      ((uint32_t)buf[9 * i + 5] << 4) |
                      ((uint32_t)buf[9 * i + 6] << 12)) & 0x3ffffu;
        uint32_t t3 = (((uint32_t)buf[9 * i + 6] >> 6) |
                      ((uint32_t)buf[9 * i + 7] << 2) |
                      ((uint32_t)buf[9 * i + 8] << 10)) & 0x3ffffu;
        dst[4 * i + 0] = PARAM_GAMMA1 - (int32_t)t0;
        dst[4 * i + 1] = PARAM_GAMMA1 - (int32_t)t1;
        dst[4 * i + 2] = PARAM_GAMMA1 - (int32_t)t2;
        dst[4 * i + 3] = PARAM_GAMMA1 - (int32_t)t3;
    }
#elif PARAM_GAMMA1 == (1 << 19)
    for (int i = lane; i < PARAM_N / 2; i += WP_SIGN_WARP_SIZE) {
        uint32_t t0 = ((uint32_t)buf[5 * i + 0] |
                      ((uint32_t)buf[5 * i + 1] << 8) |
                      ((uint32_t)buf[5 * i + 2] << 16)) & 0xfffffu;
        uint32_t t1 = (((uint32_t)buf[5 * i + 2] >> 4) |
                      ((uint32_t)buf[5 * i + 3] << 4) |
                      ((uint32_t)buf[5 * i + 4] << 12)) & 0xfffffu;
        dst[2 * i + 0] = PARAM_GAMMA1 - (int32_t)t0;
        dst[2 * i + 1] = PARAM_GAMMA1 - (int32_t)t1;
    }
#endif
#else
    stream256_state state;
    if (lane == 0) {
        aigis_shake256_gamma1_init(&state, seed, nonce);
    }
    __syncwarp();
    for (int blk = 0; blk < POLY_UNIFORM_GAMMA1_NBLOCKS; ++blk) {
        int tail = (blk * STREAM256_BLOCKBYTES) % 5;
        int avail = tail + STREAM256_BLOCKBYTES;
        int groups = avail / 5;
        int produced = (blk * STREAM256_BLOCKBYTES - tail) / 5;
        int todo = groups;
        if (produced + todo > PARAM_N / 2)
            todo = PARAM_N / 2 - produced;

        if (lane == 0)
            stream256_squeezeblocks(buf, 1, &state);
        __syncwarp();

        for (int i = lane; i < todo; i += WP_SIGN_WARP_SIZE) {
            unsigned int pos = 5u * (unsigned int)i;
            uint8_t b0 = (pos + 0 < (unsigned int)tail)
                ? buf[STREAM256_BLOCKBYTES + pos + 0]
                : buf[pos + 0 - tail];
            uint8_t b1 = (pos + 1 < (unsigned int)tail)
                ? buf[STREAM256_BLOCKBYTES + pos + 1]
                : buf[pos + 1 - tail];
            uint8_t b2 = (pos + 2 < (unsigned int)tail)
                ? buf[STREAM256_BLOCKBYTES + pos + 2]
                : buf[pos + 2 - tail];
            uint8_t b3 = (pos + 3 < (unsigned int)tail)
                ? buf[STREAM256_BLOCKBYTES + pos + 3]
                : buf[pos + 3 - tail];
            uint8_t b4 = (pos + 4 < (unsigned int)tail)
                ? buf[STREAM256_BLOCKBYTES + pos + 4]
                : buf[pos + 4 - tail];
            uint32_t t0  = b0;
            t0 |= (uint32_t)b1 << 8;
            t0 |= (uint32_t)b2 << 16;
            uint32_t t1  = b2 >> 4;
            t1 |= (uint32_t)b3 << 4;
            t1 |= (uint32_t)b4 << 12;
            t0 &= 0x3ffffu;
            t1 &= 0x3ffffu;
            int out = produced + i;
            dst[2 * out + 0] = PARAM_Q + PARAM_GAMMA1 - 1 - (int32_t)t0;
            dst[2 * out + 1] = PARAM_Q + PARAM_GAMMA1 - 1 - (int32_t)t1;
        }
        __syncwarp();

        if (lane == 0 && blk + 1 < POLY_UNIFORM_GAMMA1_NBLOCKS) {
            int used = groups * 5;
            int new_tail = avail - used;
            for (int t = 0; t < new_tail; ++t) {
                int pos = used + t;
                buf[STREAM256_BLOCKBYTES + t] = (pos < tail)
                    ? buf[STREAM256_BLOCKBYTES + pos]
                    : buf[pos - tail];
            }
        }
        __syncwarp();
    }
#endif
    __syncwarp();
}

static __device__ __noinline__ void wp_sign_pack_z_soa(
    uint8_t *sig_soa, int inst, int N, unsigned int off, const coeff_t *a, int lane)
{
#if PARAM_GAMMA1 == (1 << 17)
    for (int i = lane; i < PARAM_N / 4; i += WP_SIGN_WARP_SIZE) {
        uint32_t t0 = (uint32_t)(Z_BIAS - a[4 * i + 0]); Z_FIXUP(t0);
        uint32_t t1 = (uint32_t)(Z_BIAS - a[4 * i + 1]); Z_FIXUP(t1);
        uint32_t t2 = (uint32_t)(Z_BIAS - a[4 * i + 2]); Z_FIXUP(t2);
        uint32_t t3 = (uint32_t)(Z_BIAS - a[4 * i + 3]); Z_FIXUP(t3);
        wp_sign_store_sig(sig_soa, inst, N, off + 9 * i + 0, (uint8_t)t0);
        wp_sign_store_sig(sig_soa, inst, N, off + 9 * i + 1, (uint8_t)(t0 >> 8));
        wp_sign_store_sig(sig_soa, inst, N, off + 9 * i + 2, (uint8_t)((t0 >> 16) | (t1 << 2)));
        wp_sign_store_sig(sig_soa, inst, N, off + 9 * i + 3, (uint8_t)(t1 >> 6));
        wp_sign_store_sig(sig_soa, inst, N, off + 9 * i + 4, (uint8_t)((t1 >> 14) | (t2 << 4)));
        wp_sign_store_sig(sig_soa, inst, N, off + 9 * i + 5, (uint8_t)(t2 >> 4));
        wp_sign_store_sig(sig_soa, inst, N, off + 9 * i + 6, (uint8_t)((t2 >> 12) | (t3 << 6)));
        wp_sign_store_sig(sig_soa, inst, N, off + 9 * i + 7, (uint8_t)(t3 >> 2));
        wp_sign_store_sig(sig_soa, inst, N, off + 9 * i + 8, (uint8_t)(t3 >> 10));
    }
#elif PARAM_GAMMA1 == (1 << 19)
    for (int i = lane; i < PARAM_N / 2; i += WP_SIGN_WARP_SIZE) {
        uint32_t t0 = (uint32_t)(Z_BIAS - a[2 * i + 0]); Z_FIXUP(t0);
        uint32_t t1 = (uint32_t)(Z_BIAS - a[2 * i + 1]); Z_FIXUP(t1);
        wp_sign_store_sig(sig_soa, inst, N, off + 5 * i + 0, (uint8_t)t0);
        wp_sign_store_sig(sig_soa, inst, N, off + 5 * i + 1, (uint8_t)(t0 >> 8));
        wp_sign_store_sig(sig_soa, inst, N, off + 5 * i + 2, (uint8_t)((t0 >> 16) | (t1 << 4)));
        wp_sign_store_sig(sig_soa, inst, N, off + 5 * i + 3, (uint8_t)(t1 >> 4));
        wp_sign_store_sig(sig_soa, inst, N, off + 5 * i + 4, (uint8_t)(t1 >> 12));
    }
#endif
}

static __device__ __forceinline__ int32_t wp_sign_get_w1_hi(
    const uint8_t *packed, int k, int j)
{
    const uint8_t *r = packed + (size_t)k * POLYW1_PACKEDBYTES;
#if PARAM_GAMMA2 == (PARAM_Q - 1) / 88
    int g = j >> 2;
    int p = j & 3;
    uint8_t b0 = r[3 * g + 0];
    uint8_t b1 = r[3 * g + 1];
    uint8_t b2 = r[3 * g + 2];
    if (p == 0) return (int32_t)(b0 & 0x3fu);
    if (p == 1) return (int32_t)(((b0 >> 6) | ((b1 & 0x0fu) << 2)) & 0x3fu);
    if (p == 2) return (int32_t)(((b1 >> 4) | ((b2 & 0x03u) << 4)) & 0x3fu);
    return (int32_t)((b2 >> 2) & 0x3fu);
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 32
    uint8_t b = r[j >> 1];
    return (int32_t)((j & 1) ? (b >> 4) : (b & 0x0fu));
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 12
    int g = j >> 3;
    int p = j & 7;
    uint8_t b0 = r[3 * g + 0];
    uint8_t b1 = r[3 * g + 1];
    uint8_t b2 = r[3 * g + 2];
    if (p == 0) return (int32_t)(b0 & 0x07u);
    if (p == 1) return (int32_t)((b0 >> 3) & 0x07u);
    if (p == 2) return (int32_t)(((b0 >> 6) | ((b1 & 0x01u) << 2)) & 0x07u);
    if (p == 3) return (int32_t)((b1 >> 1) & 0x07u);
    if (p == 4) return (int32_t)((b1 >> 4) & 0x07u);
    if (p == 5) return (int32_t)(((b1 >> 7) | ((b2 & 0x03u) << 1)) & 0x07u);
    if (p == 6) return (int32_t)((b2 >> 2) & 0x07u);
    return (int32_t)((b2 >> 5) & 0x07u);
#else
    return 0;
#endif
}

static __device__ __noinline__ void wp_sign_pack_w1_poly_from_tmp(
    uint8_t *r, const coeff_t *hi, int lane)
{
#if PARAM_GAMMA2 == (PARAM_Q - 1) / 88
    for (int i = lane; i < PARAM_N / 4; i += WP_SIGN_WARP_SIZE) {
        uint32_t a0 = (uint32_t)hi[4 * i + 0];
        uint32_t a1 = (uint32_t)hi[4 * i + 1];
        uint32_t a2 = (uint32_t)hi[4 * i + 2];
        uint32_t a3 = (uint32_t)hi[4 * i + 3];
        r[3 * i + 0] = (uint8_t)(a0 | (a1 << 6));
        r[3 * i + 1] = (uint8_t)((a1 >> 2) | (a2 << 4));
        r[3 * i + 2] = (uint8_t)((a2 >> 4) | (a3 << 2));
    }
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 32
    for (int i = lane; i < PARAM_N / 2; i += WP_SIGN_WARP_SIZE) {
        uint32_t a0 = (uint32_t)hi[2 * i + 0];
        uint32_t a1 = (uint32_t)hi[2 * i + 1];
        r[i] = (uint8_t)(a0 | (a1 << 4));
    }
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 12
    for (int i = lane; i < PARAM_N / 8; i += WP_SIGN_WARP_SIZE) {
        uint32_t a0 = (uint32_t)hi[8 * i + 0];
        uint32_t a1 = (uint32_t)hi[8 * i + 1];
        uint32_t a2 = (uint32_t)hi[8 * i + 2];
        uint32_t a3 = (uint32_t)hi[8 * i + 3];
        uint32_t a4 = (uint32_t)hi[8 * i + 4];
        uint32_t a5 = (uint32_t)hi[8 * i + 5];
        uint32_t a6 = (uint32_t)hi[8 * i + 6];
        uint32_t a7 = (uint32_t)hi[8 * i + 7];
        r[3 * i + 0] = (uint8_t)(a0 | (a1 << 3) | (a2 << 6));
        r[3 * i + 1] = (uint8_t)((a2 >> 2) | (a3 << 1) |
                                  (a4 << 4) | (a5 << 7));
        r[3 * i + 2] = (uint8_t)((a5 >> 1) | (a6 << 2) | (a7 << 5));
    }
#endif
    __syncwarp();
}

static __device__ __noinline__ void wp_sign_prepare_uncached(
    wp_sign_smem_t *s,
    const uint8_t *msg, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *rnd,
    const precomp_t *pc,
    int lane)
{
    if (lane == 0) {
        keccak_state state;
#if ALGORITHM == ALGO_MLDSA
        shake256_init(&state);
        shake256_absorb(&state, pc->tr, TRBYTES);
        shake256_absorb(&state, pre, prelen);
        shake256_absorb(&state, msg, mlen);
        shake256_finalize(&state);
        shake256_squeeze(s->mu, CRHBYTES, &state);

        shake256_init(&state);
        shake256_absorb(&state, pc->key, SEEDBYTES);
#if RNDBYTES > 0
        shake256_absorb(&state, rnd, RNDBYTES);
#endif
        shake256_absorb(&state, s->mu, CRHBYTES);
        shake256_finalize(&state);
        shake256_squeeze(s->seed, CRHBYTES, &state);
#else
        shake256_init(&state);
        shake256_absorb(&state, pc->tr, TRBYTES);
        shake256_absorb(&state, msg, mlen);
        shake256_finalize(&state);
        shake256_squeeze(s->mu, CRHBYTES, &state);

        for (int i = 0; i < SEEDBYTES; ++i) s->seed[i] = pc->key[i];
        for (int i = 0; i < CRHBYTES; ++i) s->seed[SEEDBYTES + i] = s->mu[i];
#endif
    }
    __syncwarp();
}

static __device__ __noinline__ void wp_sign_prepare_cached(
    wp_sign_smem_t *s, const uint8_t *cache_raw, int lane)
{
    if (lane == 0) {
        const wp_sign_cache_t *cache = (const wp_sign_cache_t *)cache_raw;
        for (int i = 0; i < CRHBYTES; ++i) s->mu[i] = cache->mu[i];
#if ALGORITHM == ALGO_MLDSA
        for (int i = 0; i < CRHBYTES; ++i) s->seed[i] = cache->rhoprime[i];
#else
        for (int i = 0; i < SEEDBYTES + CRHBYTES; ++i) s->seed[i] = cache->key_mu[i];
#endif
    }
    __syncwarp();
}

static __device__ __noinline__ void wp_sign_matrix_y(
    wp_sign_smem_t *s, const precomp_t *pc, uint16_t nonce_base, int lane)
{
    for (int k = 0; k < PARAM_K; ++k)
        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE)
            s->w[(size_t)k * PARAM_N + j] = 0;
    __syncwarp();

    for (int l = 0; l < PARAM_L; ++l) {
        coeff_t *yl = s->y + (size_t)l * PARAM_N;
        wp_sign_sample_y_poly(yl, s->seed, GAMMA1_NONCE(nonce_base, l), lane, s->work);

        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE)
            s->tmp[j] = yl[j];
        __syncwarp();
        ntt_warp_par(s->tmp, lane);

        for (int k = 0; k < PARAM_K; ++k) {
            coeff_t *wk = s->w + (size_t)k * PARAM_N;
            const coeff_t *akl = pc->mat[k].vec[l].coeffs;
            for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE) {
                coeff_t prod = montgomery_reduce((coeff2_t)akl[j] * s->tmp[j]);
                wk[j] += prod;
            }
        }
        __syncwarp();
    }

    for (int k = 0; k < PARAM_K; ++k) {
        coeff_t *wk = s->w + (size_t)k * PARAM_N;
        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE) {
#if ALGORITHM == ALGO_MLDSA
            wk[j] = reduce32(wk[j]);
#else
            wk[j] = barrat_reduce(wk[j]);
#endif
        }
        __syncwarp();
        invntt_warp_par(wk, lane);
        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE) {
#if ALGORITHM == ALGO_MLDSA
            int32_t a = caddq(reduce32(wk[j]));
            int32_t lo;
            int32_t hi = decompose(&lo, a);
            wk[j] = lo;
            s->tmp[j] = hi;
#else
            int32_t a = freeze2q(wk[j]);
            int32_t lo;
            int32_t hi = decompose(&lo, a);
            wk[j] = a;
            s->tmp[j] = hi;
#endif
        }
        __syncwarp();
        wp_sign_pack_w1_poly_from_tmp(
            s->packed_w1 + (size_t)k * POLYW1_PACKEDBYTES, s->tmp, lane);
    }
}

static __device__ __noinline__ void wp_sign_make_challenge(
    wp_sign_smem_t *s, uint8_t *sig_soa, int inst, int N, int lane)
{
    if (lane == 0) {
#if ALGORITHM == ALGO_MLDSA
        keccak_state state;
        shake256_init(&state);
        shake256_absorb(&state, s->mu, CRHBYTES);
        shake256_absorb(&state, s->packed_w1, PARAM_K * POLYW1_PACKEDBYTES);
        shake256_finalize(&state);
        shake256_squeeze(s->work, CTILDEBYTES, &state);
        for (unsigned int i = 0; i < CTILDEBYTES; ++i)
            wp_sign_store_sig(sig_soa, inst, N, i, s->work[i]);
        poly_challenge((poly *)s->cp, s->work);
#else
        poly_challenge((poly *)s->cp, s->mu, s->packed_w1,
                       PARAM_K * POLYW1_PACKEDBYTES);
        unsigned int offset = PARAM_L * POLYZ_PACKEDBYTES + PARAM_OMEGA + PARAM_K;
        uint64_t signs = 0;
        uint64_t mask = 1;
        for (unsigned int i = 0; i < PARAM_N / 8; ++i) {
            uint8_t b = 0;
            for (unsigned int j = 0; j < 8; ++j) {
                coeff_t c = s->cp[8 * i + j];
                if (c != 0) {
                    b |= (uint8_t)(1u << j);
                    if (c == (PARAM_Q - 1)) signs |= mask;
                    mask <<= 1;
                }
            }
            wp_sign_store_sig(sig_soa, inst, N, offset + i, b);
        }
        offset += PARAM_N / 8;
        for (unsigned int i = 0; i < 8; ++i)
            wp_sign_store_sig(sig_soa, inst, N, offset + i, (uint8_t)(signs >> (8 * i)));
#endif
    }
    __syncwarp();
    ntt_warp_par(s->cp, lane);
}

static __device__ __noinline__ int wp_sign_check_s2(
    wp_sign_smem_t *s, const precomp_t *pc, int lane)
{
    for (int k = 0; k < PARAM_K; ++k) {
        const coeff_t *sk = pc->s2_ntt.vec[k].coeffs;
        coeff_t *wk = s->w + (size_t)k * PARAM_N;
        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE)
            s->tmp[j] = montgomery_reduce((coeff2_t)s->cp[j] * sk[j]);
        __syncwarp();
        invntt_warp_par(s->tmp, lane);

        int bad = 0;
        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE) {
#if ALGORITHM == ALGO_MLDSA
            int32_t v = reduce32(wk[j] - s->tmp[j]);
            wk[j] = v;
            bad |= wp_sign_coeff_chknorm(v, PARAM_GAMMA2 - PARAM_BETA2);
#else
            int32_t v = freeze4q(wk[j] - s->tmp[j]);
            int32_t lo;
            int32_t hi = decompose(&lo, v);
            lo = freeze2q(lo);
            wk[j] = v;
            bad |= (hi != wp_sign_get_w1_hi(s->packed_w1, k, j));
            bad |= wp_sign_coeff_chknorm(lo, PARAM_GAMMA2 - PARAM_BETA2);
#endif
        }
        if (wp_sign_any(bad)) return 1;
        __syncwarp();
    }
    return 0;
}

static __device__ __noinline__ int wp_sign_check_pack_z(
    wp_sign_smem_t *s, const precomp_t *pc, uint8_t *sig_soa, int inst, int N, int lane)
{
    for (int l = 0; l < PARAM_L; ++l) {
        const coeff_t *sl = pc->s1_ntt.vec[l].coeffs;
        const coeff_t *yl = s->y + (size_t)l * PARAM_N;
        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE)
            s->tmp[j] = montgomery_reduce((coeff2_t)s->cp[j] * sl[j]);
        __syncwarp();
        invntt_warp_par(s->tmp, lane);

        int bad = 0;
        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE) {
#if ALGORITHM == ALGO_MLDSA
            int32_t z = reduce32(s->tmp[j] + yl[j]);
#else
            int32_t z = freeze4q(s->tmp[j] + yl[j]);
#endif
            s->tmp[j] = z;
            bad |= wp_sign_coeff_chknorm(z, PARAM_GAMMA1 - PARAM_BETA1);
        }
        if (wp_sign_any(bad)) return 1;

#if ALGORITHM == ALGO_MLDSA
        unsigned int off = CTILDEBYTES + (unsigned int)l * POLYZ_PACKEDBYTES;
#else
        unsigned int off = (unsigned int)l * POLYZ_PACKEDBYTES;
#endif
        wp_sign_pack_z_soa(sig_soa, inst, N, off, s->tmp, lane);
        __syncwarp();
    }
    return 0;
}

static __device__ __noinline__ int wp_sign_check_t0_accumulate(
    wp_sign_smem_t *s, const precomp_t *pc,
    uint8_t *sig_soa, int inst, int N, int lane)
{
#if ALGORITHM == ALGO_AIGIS
    const unsigned int hint_off = PARAM_L * POLYZ_PACKEDBYTES;
    for (unsigned int i = lane; i < PARAM_OMEGA + PARAM_K; i += WP_SIGN_WARP_SIZE)
        wp_sign_store_sig(sig_soa, inst, N, hint_off + i, 0);
    __syncwarp();

    unsigned int hint_count = 0;
    int hint_overflow = 0;
#endif

    for (int k = 0; k < PARAM_K; ++k) {
        const coeff_t *tk = pc->t0_ntt.vec[k].coeffs;
        coeff_t *wk = s->w + (size_t)k * PARAM_N;
        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE)
            s->tmp[j] = montgomery_reduce((coeff2_t)s->cp[j] * tk[j]);
        __syncwarp();
        invntt_warp_par(s->tmp, lane);

        int bad = 0;
        for (int j = lane; j < PARAM_N; j += WP_SIGN_WARP_SIZE) {
#if ALGORITHM == ALGO_MLDSA
            int32_t ct0 = reduce32(s->tmp[j]);
            bad |= wp_sign_coeff_chknorm(ct0, PARAM_GAMMA2);
            wk[j] = wk[j] + ct0;
#else
            int32_t ct0 = freeze2q(s->tmp[j]);
            bad |= wp_sign_coeff_chknorm(ct0, PARAM_GAMMA2);
            wk[j] = freeze2q(wk[j] + ct0);
#endif
        }
        if (wp_sign_any(bad)) return 1;
        __syncwarp();
#if ALGORITHM == ALGO_AIGIS
        if (lane == 0) {
            for (unsigned int j = 0; j < PARAM_N; ++j) {
                int32_t ct0 = freeze2q(s->tmp[j]);
                int h = make_hint(wk[j], 2 * PARAM_Q - ct0);
                if (h) {
                    if (hint_count < PARAM_OMEGA)
                        wp_sign_store_sig(sig_soa, inst, N,
                                          hint_off + hint_count, (uint8_t)j);
                    hint_count++;
                }
            }
            if (hint_count <= PARAM_OMEGA)
                wp_sign_store_sig(sig_soa, inst, N,
                                  hint_off + PARAM_OMEGA + k,
                                  (uint8_t)hint_count);
        }
        __syncwarp();
#endif
    }
#if ALGORITHM == ALGO_AIGIS
    if (lane == 0)
        hint_overflow = (hint_count > PARAM_OMEGA);
    hint_overflow = __shfl_sync(0xffffffffull, hint_overflow, 0);
    if (hint_overflow) return 2;
#endif
    return 0;
}

static __device__ __noinline__ int wp_sign_pack_hints(
    wp_sign_smem_t *s, uint8_t *sig_soa, int inst, int N, int lane)
{
#if ALGORITHM == ALGO_MLDSA
    const unsigned int hint_off = CTILDEBYTES + PARAM_L * POLYZ_PACKEDBYTES;
    for (unsigned int i = lane; i < PARAM_OMEGA + PARAM_K; i += WP_SIGN_WARP_SIZE)
        wp_sign_store_sig(sig_soa, inst, N, hint_off + i, 0);
    __syncwarp();

    unsigned int count = 0;
    int overflow = 0;
    if (lane == 0) {
        for (unsigned int k = 0; k < PARAM_K; ++k) {
            coeff_t *wk = s->w + (size_t)k * PARAM_N;
            for (unsigned int j = 0; j < PARAM_N; ++j) {
                int h = make_hint(wk[j], wp_sign_get_w1_hi(s->packed_w1, k, j));
                if (h) {
                    if (count < PARAM_OMEGA)
                        wp_sign_store_sig(sig_soa, inst, N, hint_off + count, (uint8_t)j);
                    count++;
                }
            }
            if (count <= PARAM_OMEGA)
                wp_sign_store_sig(sig_soa, inst, N,
                                  hint_off + PARAM_OMEGA + k, (uint8_t)count);
        }
        overflow = (count > PARAM_OMEGA);
    }
    overflow = __shfl_sync(0xffffffffull, overflow, 0);
    return overflow;
#else
    return 0;
#endif
}

static __device__ __noinline__ int wp_sign_core(
    uint8_t *sig_soa, size_t *siglen_arr,
    const uint8_t *msg, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *rnd,
    const uint8_t *cache_raw,
    const precomp_t *pc,
    int *results, int N, int inst, int cached,
    unsigned long long *stats,
    wp_sign_smem_t *s,
    int lane)
{
    if (cached)
        wp_sign_prepare_cached(s, cache_raw, lane);
    else
        wp_sign_prepare_uncached(s, msg, mlen, pre, prelen, rnd, pc, lane);

#if BATCH_SIGN_NONCE_DIVERSIFY
#if ALGORITHM == ALGO_AIGIS
    uint16_t nonce = (uint16_t)(((unsigned int)inst * PARAM_L) & 0xffffu);
#else
    uint16_t nonce = (uint16_t)inst;
#endif
#else
    uint16_t nonce = 0;
#endif

    for (;;) {
        uint16_t nonce_base = nonce;
#if ALGORITHM == ALGO_AIGIS
        nonce = (uint16_t)(nonce + PARAM_L);
#else
        nonce = (uint16_t)(nonce + 1);
#endif
        if (lane == 0 && stats) atomicAdd(&stats[WP_SIGN_STAT_ATTEMPTS], 1ull);

        wp_sign_matrix_y(s, pc, nonce_base, lane);
        wp_sign_make_challenge(s, sig_soa, inst, N, lane);

        if (wp_sign_check_s2(s, pc, lane)) {
            if (lane == 0 && stats) atomicAdd(&stats[WP_SIGN_STAT_REJ_S2], 1ull);
            continue;
        }
        if (wp_sign_check_pack_z(s, pc, sig_soa, inst, N, lane)) {
            if (lane == 0 && stats) atomicAdd(&stats[WP_SIGN_STAT_REJ_Z], 1ull);
            continue;
        }
        int t0_status = wp_sign_check_t0_accumulate(s, pc, sig_soa, inst, N, lane);
        if (t0_status) {
            if (lane == 0 && stats) {
                atomicAdd(&stats[(t0_status == 2)
                                 ? WP_SIGN_STAT_REJ_HINT
                                 : WP_SIGN_STAT_REJ_T0], 1ull);
            }
            continue;
        }
#if ALGORITHM == ALGO_MLDSA
        if (wp_sign_pack_hints(s, sig_soa, inst, N, lane)) {
            if (lane == 0 && stats) atomicAdd(&stats[WP_SIGN_STAT_REJ_HINT], 1ull);
            continue;
        }
#endif

        if (lane == 0) {
            siglen_arr[inst] = CRYPTO_BYTES;
            results[inst] = 0;
            if (stats) atomicAdd(&stats[WP_SIGN_STAT_OK], 1ull);
        }
        return 0;
    }
}

__global__ void __launch_bounds__(WP_SIGN_TPB, 1)
kernel_batch_sign_warp_precomp(
    uint8_t *sig_soa, size_t *siglen_arr,
    const uint8_t *msg, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *rnd,
    const precomp_t *pc,
    int *results, int N, int base_idx,
    unsigned long long *stats)
{
    extern __shared__ unsigned char smem[];
    int lane = threadIdx.x & (WP_SIGN_WARP_SIZE - 1);
    int warp_slot = threadIdx.x >> 5;
    int inst = base_idx + (int)blockIdx.x * WP_SIGN_WARPS_BLOCK + warp_slot;
    if (inst >= N) return;

    wp_sign_smem_t s;
    wp_sign_smem_init(&s, smem, warp_slot);
    wp_sign_core(sig_soa, siglen_arr, msg, mlen, pre, prelen, rnd, NULL,
                 pc, results, N, inst, 0, stats, &s, lane);
}

__global__ void __launch_bounds__(WP_SIGN_TPB, 1)
kernel_batch_sign_warp_precomp_cached(
    uint8_t *sig_soa, size_t *siglen_arr,
    const uint8_t *cache_raw,
    const precomp_t *pc,
    int *results, int N, int base_idx,
    unsigned long long *stats)
{
    extern __shared__ unsigned char smem[];
    int lane = threadIdx.x & (WP_SIGN_WARP_SIZE - 1);
    int warp_slot = threadIdx.x >> 5;
    int inst = base_idx + (int)blockIdx.x * WP_SIGN_WARPS_BLOCK + warp_slot;
    if (inst >= N) return;

    wp_sign_smem_t s;
    wp_sign_smem_init(&s, smem, warp_slot);
    wp_sign_core(sig_soa, siglen_arr, NULL, 0, NULL, 0, NULL, cache_raw,
                 pc, results, N, inst, 1, stats, &s, lane);
}

__global__ void kernel_wp_sign_sig_soa_to_aos(
    uint8_t *sig_aos, const uint8_t *sig_soa, int N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * CRYPTO_BYTES;
    if (idx >= total) return;
    int inst = (int)(idx / CRYPTO_BYTES);
    int byte = (int)(idx - (size_t)inst * CRYPTO_BYTES);
    sig_aos[idx] = sig_soa[(size_t)byte * (size_t)N + (size_t)inst];
}

static inline hipError_t batch_sign_warp_set_smem_attributes(void) {
    size_t smem = batch_sign_warp_smem_bytes();
    hipError_t e = hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_batch_sign_warp_precomp),
                                         hipFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)smem);
    if (e != hipSuccess) return e;
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_batch_sign_warp_precomp_cached),
                                hipFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem);
}

#endif /* BATCH_SIGN_WARP_CUH */
