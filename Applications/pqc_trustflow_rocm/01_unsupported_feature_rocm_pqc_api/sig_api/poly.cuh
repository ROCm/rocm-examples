#ifndef POLY_CUH
#define POLY_CUH

#include <stdint.h>
#include "params.h"
#include "ntt.cuh"
#include "reduce.cuh"
#include "rounding.cuh"
#include "symmetric.cuh"

typedef struct { int32_t coeffs[PARAM_N]; } poly;

/* ==== Basic arithmetic ==== */
static __device__ void poly_reduce(poly *a) {
  for (unsigned int i = 0; i < PARAM_N; ++i) a->coeffs[i] = reduce32(a->coeffs[i]);
}
static __device__ void poly_caddq(poly *a) {
  for (unsigned int i = 0; i < PARAM_N; ++i) a->coeffs[i] = caddq(a->coeffs[i]);
}
static __device__ void poly_freeze2q(poly *a) {
  for (unsigned int i = 0; i < PARAM_N; ++i) a->coeffs[i] = freeze2q(a->coeffs[i]);
}
static __device__ void poly_freeze4q(poly *a) {
  for (unsigned int i = 0; i < PARAM_N; ++i) a->coeffs[i] = freeze4q(a->coeffs[i]);
}
static __device__ void poly_add(poly *c, const poly *a, const poly *b) {
  for (unsigned int i = 0; i < PARAM_N; ++i) c->coeffs[i] = a->coeffs[i] + b->coeffs[i];
}
/* Unified sub: ML-DSA (COEFF_BIAS=0) → a-b; Aigis (COEFF_BIAS=Q) → a+2Q-b */
static __device__ void poly_sub(poly *c, const poly *a, const poly *b) {
  for (unsigned int i = 0; i < PARAM_N; ++i) c->coeffs[i] = a->coeffs[i] + 2 * COEFF_BIAS - b->coeffs[i];
}
#if ALGORITHM == ALGO_AIGIS
static __device__ void poly_neg(poly *a) {
  for (unsigned int i = 0; i < PARAM_N; ++i) a->coeffs[i] = 2 * PARAM_Q - a->coeffs[i];
}
#endif
static __device__ void poly_shiftl(poly *a) {
  for (unsigned int i = 0; i < PARAM_N; ++i) a->coeffs[i] <<= PARAM_D;
}
static __device__ void poly_ntt(poly *a) { ntt(a->coeffs); }
static __device__ void poly_invntt_tomont(poly *a) { invntt_tomont(a->coeffs); }
static __device__ void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
  for (unsigned int i = 0; i < PARAM_N; ++i)
    c->coeffs[i] = montgomery_reduce((int64_t)a->coeffs[i] * b->coeffs[i]);
}
static __device__ void poly_power2round(poly *a1, poly *a0, const poly *a) {
  for (unsigned int i = 0; i < PARAM_N; ++i)
    a1->coeffs[i] = power2round(&a0->coeffs[i], a->coeffs[i]);
}
static __device__ void poly_decompose(poly *a1, poly *a0, const poly *a) {
  for (unsigned int i = 0; i < PARAM_N; ++i)
    a1->coeffs[i] = decompose(&a0->coeffs[i], a->coeffs[i]);
}
static __device__ unsigned int poly_make_hint(poly *h, const poly *a0, const poly *a1) {
  unsigned int s = 0;
  for (unsigned int i = 0; i < PARAM_N; ++i) {
    h->coeffs[i] = make_hint(a0->coeffs[i], a1->coeffs[i]);
    s += h->coeffs[i];
  }
  return s;
}
static __device__ void poly_use_hint(poly *b, const poly *a, const poly *h) {
  for (unsigned int i = 0; i < PARAM_N; ++i)
    b->coeffs[i] = use_hint(a->coeffs[i], h->coeffs[i]);
}

/* ---- chknorm: check if any coeff has |coeff| >= B ---- */
#if ALGORITHM == ALGO_MLDSA
static __device__ int poly_chknorm(const poly *a, int32_t B) {
  if (B > (PARAM_Q - 1) / 8) return 1;
  for (unsigned int i = 0; i < PARAM_N; ++i) {
    int32_t t = a->coeffs[i] >> 31;
    t = a->coeffs[i] - (t & 2 * a->coeffs[i]);
    if (t >= B) return 1;
  }
  return 0;
}
#elif ALGORITHM == ALGO_AIGIS
/* Aigis: unsigned coeff ∈ [0,Q), distance = |(Q-1)/2 - coeff| */
static __device__ int poly_chknorm(const poly *a, int32_t B) {
  for (unsigned int i = 0; i < PARAM_N; ++i) {
    int32_t t = (PARAM_Q - 1) / 2 - a->coeffs[i];
    t ^= (t >> 31);
    t = (PARAM_Q - 1) / 2 - t;
    if (t >= B) return 1;
  }
  return 0;
}
#endif

/* ================================================================
 *  Uniform rejection sampling for matrix A
 * ================================================================ */
static __device__ unsigned int rej_uniform(int32_t *a, unsigned int len,
                                            const uint8_t *buf, unsigned int buflen) {
  unsigned int ctr = 0, pos = 0;
  while (ctr < len && pos + 3 <= buflen) {
    uint32_t t = buf[pos++] | ((uint32_t)buf[pos++] << 8) | ((uint32_t)buf[pos++] << 16);
    t &= (1u << PARAM_QBITS) - 1;
    if (t < (uint32_t)PARAM_Q) a[ctr++] = (int32_t)t;
  }
  return ctr;
}

#define POLY_UNIFORM_NBLOCKS ((768 + STREAM128_BLOCKBYTES - 1) / STREAM128_BLOCKBYTES)

/* Unified poly_uniform: 共享函数体, 仅 stream init 按算法分流 */
static __device__ __noinline__ void poly_uniform(poly *a, const uint8_t seed[SEEDBYTES], uint16_t nonce) {
  unsigned int ctr, off;
  unsigned int buflen = POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES + 2];
  stream128_state state;
#if ALGORITHM == ALGO_MLDSA
  stream128_init(&state, seed, nonce);
#elif ALGORITHM == ALGO_AIGIS
  aigis_shake128_stream_init(&state, seed, (uint8_t)nonce);
#endif
  stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);
  ctr = rej_uniform(a->coeffs, PARAM_N, buf, buflen);
  while (ctr < PARAM_N) {
    off = buflen % 3;
    for (unsigned int i = 0; i < off; ++i) buf[i] = buf[buflen - off + i];
    stream128_squeezeblocks(buf + off, 1, &state);
    buflen = STREAM128_BLOCKBYTES + off;
    ctr += rej_uniform(a->coeffs + ctr, PARAM_N - ctr, buf, buflen);
  }
}

static __device__ __noinline__ void poly_uniform_to(coeff_t *a, const uint8_t seed[SEEDBYTES], uint16_t nonce) {
  unsigned int ctr, off;
  unsigned int buflen = POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES + 2];
  stream128_state state;
#if ALGORITHM == ALGO_MLDSA
  stream128_init(&state, seed, nonce);
#elif ALGORITHM == ALGO_AIGIS
  aigis_shake128_stream_init(&state, seed, (uint8_t)nonce);
#endif
  stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);
  ctr = rej_uniform(a, PARAM_N, buf, buflen);
  while (ctr < PARAM_N) {
    off = buflen % 3;
    for (unsigned int i = 0; i < off; ++i) buf[i] = buf[buflen - off + i];
    stream128_squeezeblocks(buf + off, 1, &state);
    buflen = STREAM128_BLOCKBYTES + off;
    ctr += rej_uniform(a + ctr, PARAM_N - ctr, buf, buflen);
  }
}

/* ================================================================
 *  Eta rejection sampling (s1 and s2)
 *  rej 函数按算法分流, poly_uniform_eta 共享骨架
 * ================================================================ */

#if ALGORITHM == ALGO_MLDSA
static __device__ unsigned int rej_eta_mldsa_to(int32_t *a, unsigned int len,
                                                 const uint8_t *buf,
                                                 unsigned int buflen,
                                                 int eta) {
  unsigned int ctr = 0, pos = 0;
  while (ctr < len && pos < buflen) {
    uint32_t t0 = buf[pos] & 0x0F;
    uint32_t t1 = buf[pos++] >> 4;
    if (eta == 2) {
      if (t0 < 15) {
        t0 = t0 - (205 * t0 >> 10) * 5;
        a[ctr++] = 2 - (int32_t)t0;
      }
      if (t1 < 15 && ctr < len) {
        t1 = t1 - (205 * t1 >> 10) * 5;
        a[ctr++] = 2 - (int32_t)t1;
      }
    } else {
      if (t0 < 9) a[ctr++] = 4 - (int32_t)t0;
      if (t1 < 9 && ctr < len) a[ctr++] = 4 - (int32_t)t1;
    }
  }
  return ctr;
}

/* ML-DSA: Output CENTERED int32 in [-ETA, ETA]
 * FIPS 204 Algorithm 15 (CoeffFromHalfByte):
 *   ETA==2: accept b<15, coeff = 2 - (b mod 5)
 *   ETA==4: accept b<9,  coeff = 4 - b           */
static __device__ unsigned int rej_eta_val(int32_t *a, unsigned int len,
                                            const uint8_t *buf, unsigned int buflen) {
  unsigned int ctr = 0, pos = 0;
  while (ctr < len && pos < buflen) {
    uint32_t t0 = buf[pos] & 0x0F;
    uint32_t t1 = buf[pos++] >> 4;
#if PARAM_ETA_S1 == 2
    if (t0 < 15) {
      t0 = t0 - (205*t0 >> 10)*5;
      a[ctr++] = 2 - (int32_t)t0;
    }
    if (t1 < 15 && ctr < len) {
      t1 = t1 - (205*t1 >> 10)*5;
      a[ctr++] = 2 - (int32_t)t1;
    }
#elif PARAM_ETA_S1 == 4
    if (t0 < 9) a[ctr++] = 4 - (int32_t)t0;
    if (t1 < 9 && ctr < len) a[ctr++] = 4 - (int32_t)t1;
#endif
  }
  return ctr;
}
/* Dispatch macros for unified poly_uniform_eta */
#define rej_eta1(a, len, buf, buflen) rej_eta_val(a, len, buf, buflen)
#define rej_eta2(a, len, buf, buflen) rej_eta_val(a, len, buf, buflen)

#elif ALGORITHM == ALGO_AIGIS
/* Aigis: Output UNSIGNED Q+ETA-t
 * ETA1=1: 2-bit extraction (4 values per byte)
 * ETA1=2: 3-bit extraction (8 values per 3 bytes) — matches CPU reference
 * ETA1=3: 3-bit extraction (same structure) */
static __device__ unsigned int rej_eta1_aigis(int32_t *a, unsigned int len,
                                               const uint8_t *buf, unsigned int buflen) {
  unsigned int ctr = 0, pos = 0;
#if PARAM_ETA_S1 == 1
  while (ctr < len && pos < buflen) {
    uint32_t t0 = buf[pos] & 0x03;
    uint32_t t1 = (buf[pos] >> 2) & 0x03;
    uint32_t t2 = (buf[pos] >> 4) & 0x03;
    uint32_t t3 = (buf[pos++] >> 6) & 0x03;
    if (t0 <= 2u * PARAM_ETA_S1) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t0;
    if (t1 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t1;
    if (t2 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t2;
    if (t3 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t3;
  }
#elif PARAM_ETA_S1 == 2 || PARAM_ETA_S1 == 3
  /* 3-bit extraction: 8 values from every 3 bytes */
  while (ctr < len && pos + 3 <= buflen) {
    uint32_t t0 = buf[pos] & 0x07;
    uint32_t t1 = (buf[pos] >> 3) & 0x07;
    uint32_t t2 = (buf[pos] >> 6) | ((uint32_t)(buf[pos + 1] & 0x01) << 2);
    uint32_t t3 = (buf[pos + 1] >> 1) & 0x07;
    uint32_t t4 = (buf[pos + 1] >> 4) & 0x07;
    uint32_t t5 = (buf[pos + 1] >> 7) | ((uint32_t)(buf[pos + 2] & 0x03) << 1);
    uint32_t t6 = (buf[pos + 2] >> 2) & 0x07;
    uint32_t t7 = buf[pos + 2] >> 5;
    pos += 3;
    if (t0 <= 2u * PARAM_ETA_S1) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t0;
    if (t1 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t1;
    if (t2 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t2;
    if (t3 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t3;
    if (t4 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t4;
    if (t5 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t5;
    if (t6 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t6;
    if (t7 <= 2u * PARAM_ETA_S1 && ctr < len) a[ctr++] = PARAM_Q + PARAM_ETA_S1 - (int32_t)t7;
  }
#endif
  return ctr;
}
/* rej_eta2_aigis: exact mirror of CPU rej_eta2() — two do-while loops, returns pos (byte position) */
static __device__ unsigned int rej_eta2_aigis(int32_t *a, unsigned int len,
                                               const uint8_t *buf) {
  unsigned int ctr = 0, pos = 0;
  uint8_t t0, t1;

  /* Fast loop: no ctr check on t1 */
  do {
#if PARAM_ETA_S2 == 3
    t0 = buf[pos] & 0x07;
    t1 = buf[pos++] >> 5;
#else
    t0 = buf[pos] & 0x0F;
    t1 = buf[pos++] >> 4;
#endif
    if (t0 <= 2u * PARAM_ETA_S2)
      a[ctr++] = PARAM_Q + PARAM_ETA_S2 - (int32_t)t0;
    if (t1 <= 2u * PARAM_ETA_S2)
      a[ctr++] = PARAM_Q + PARAM_ETA_S2 - (int32_t)t1;
  } while (ctr < len - 2);

  /* Slow loop: ctr check on t1 */
  do {
#if PARAM_ETA_S2 == 3
    t0 = buf[pos] & 0x07;
    t1 = buf[pos++] >> 5;
#else
    t0 = buf[pos] & 0x0F;
    t1 = buf[pos++] >> 4;
#endif
    if (t0 <= 2u * PARAM_ETA_S2)
      a[ctr++] = PARAM_Q + PARAM_ETA_S2 - (int32_t)t0;
    if (t1 <= 2u * PARAM_ETA_S2 && ctr < len)
      a[ctr++] = PARAM_Q + PARAM_ETA_S2 - (int32_t)t1;
  } while (ctr < len);

  return pos;
}
/* Dispatch macros for unified poly_uniform_eta */
#define rej_eta1(a, len, buf, buflen) rej_eta1_aigis(a, len, buf, buflen)
#define rej_eta2(a, len, buf) rej_eta2_aigis(a, len, buf)
#endif /* rej_eta */

/* Block count 宏: match CPU reference (FIPS 204) */
#if ALGORITHM == ALGO_MLDSA
#if PARAM_ETA_S1 == 2
#define POLY_UNIFORM_ETA1_NBLOCKS ((136 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#elif PARAM_ETA_S1 == 4
#define POLY_UNIFORM_ETA1_NBLOCKS ((227 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#endif
#if PARAM_ETA_S2 == 2
#define POLY_UNIFORM_ETA2_NBLOCKS ((136 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#elif PARAM_ETA_S2 == 4
#define POLY_UNIFORM_ETA2_NBLOCKS ((227 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#endif
#elif ALGORITHM == ALGO_AIGIS
#define POLY_UNIFORM_ETA1_NBLOCKS 2
#define POLY_UNIFORM_ETA2_NBLOCKS 3
#endif

/* Unified poly_uniform_eta_s1: 共享骨架, 仅 stream init 和 rej 按算法分流 */
static __device__ __noinline__ void poly_uniform_eta_s1(poly *a,
                                                         const uint8_t *seed,
                                                         uint16_t nonce) {
  uint8_t buf[POLY_UNIFORM_ETA1_NBLOCKS * STREAM256_BLOCKBYTES];
  stream256_state state;
#if ALGORITHM == ALGO_MLDSA
  stream256_init(&state, seed, nonce);
#elif ALGORITHM == ALGO_AIGIS
  aigis_shake256_eta_init(&state, seed, (uint8_t)nonce);
#endif
  stream256_squeezeblocks(buf, POLY_UNIFORM_ETA1_NBLOCKS, &state);
  unsigned int ctr = rej_eta1(a->coeffs, PARAM_N, buf,
                               POLY_UNIFORM_ETA1_NBLOCKS * STREAM256_BLOCKBYTES);
  while (ctr < PARAM_N) {
    stream256_squeezeblocks(buf, 1, &state);
    ctr += rej_eta1(a->coeffs + ctr, PARAM_N - ctr, buf, STREAM256_BLOCKBYTES);
  }
}

static __device__ __noinline__ void poly_uniform_eta_s1_to(coeff_t *a,
                                                            const uint8_t *seed,
                                                            uint16_t nonce) {
  uint8_t buf[POLY_UNIFORM_ETA1_NBLOCKS * STREAM256_BLOCKBYTES];
  stream256_state state;
#if ALGORITHM == ALGO_MLDSA
  stream256_init(&state, seed, nonce);
#elif ALGORITHM == ALGO_AIGIS
  aigis_shake256_eta_init(&state, seed, (uint8_t)nonce);
#endif
  stream256_squeezeblocks(buf, POLY_UNIFORM_ETA1_NBLOCKS, &state);
#if ALGORITHM == ALGO_MLDSA
  unsigned int ctr = rej_eta_mldsa_to(a, PARAM_N, buf,
                                      POLY_UNIFORM_ETA1_NBLOCKS * STREAM256_BLOCKBYTES,
                                      PARAM_ETA_S1);
  while (ctr < PARAM_N) {
    stream256_squeezeblocks(buf, 1, &state);
    ctr += rej_eta_mldsa_to(a + ctr, PARAM_N - ctr, buf,
                            STREAM256_BLOCKBYTES, PARAM_ETA_S1);
  }
#else
  unsigned int ctr = rej_eta1(a, PARAM_N, buf,
                               POLY_UNIFORM_ETA1_NBLOCKS * STREAM256_BLOCKBYTES);
  while (ctr < PARAM_N) {
    stream256_squeezeblocks(buf, 1, &state);
    ctr += rej_eta1(a + ctr, PARAM_N - ctr, buf, STREAM256_BLOCKBYTES);
  }
#endif
}

static __device__ __noinline__ void poly_uniform_eta_s2(poly *a,
                                                         const uint8_t *seed,
                                                         uint16_t nonce) {
  uint8_t buf[POLY_UNIFORM_ETA2_NBLOCKS * STREAM256_BLOCKBYTES];
  stream256_state state;
#if ALGORITHM == ALGO_MLDSA
  stream256_init(&state, seed, nonce);
  stream256_squeezeblocks(buf, POLY_UNIFORM_ETA2_NBLOCKS, &state);
  unsigned int ctr = rej_eta2(a->coeffs, PARAM_N, buf,
                               POLY_UNIFORM_ETA2_NBLOCKS * STREAM256_BLOCKBYTES);
  while (ctr < PARAM_N) {
    stream256_squeezeblocks(buf, 1, &state);
    ctr += rej_eta2(a->coeffs + ctr, PARAM_N - ctr, buf, STREAM256_BLOCKBYTES);
  }
#elif ALGORITHM == ALGO_AIGIS
  aigis_shake256_eta_init(&state, seed, (uint8_t)nonce);
  stream256_squeezeblocks(buf, 2, &state);

#if PARAM_ETA_S2 == 3
  /* ETA2=3: single pass, probability of needing >2 blocks is < 2^{-378} */
  rej_eta2(a->coeffs, PARAM_N, buf);

#elif PARAM_ETA_S2 == 5
  /* ETA2=5: two-pass split at 223 — exactly mirrors CPU poly_uniform_eta2() */
  {
    unsigned int pos = rej_eta2(a->coeffs, 223, buf);

    if (2u * STREAM256_BLOCKBYTES - pos < 85) {
      stream256_squeezeblocks(buf + 2 * STREAM256_BLOCKBYTES, 1, &state);
    }

    rej_eta2(&a->coeffs[223], 33, &buf[pos]);
  }
#endif
#endif
}

static __device__ __noinline__ void poly_uniform_eta_s2_to(coeff_t *a,
                                                            const uint8_t *seed,
                                                            uint16_t nonce) {
  uint8_t buf[POLY_UNIFORM_ETA2_NBLOCKS * STREAM256_BLOCKBYTES];
  stream256_state state;
#if ALGORITHM == ALGO_MLDSA
  stream256_init(&state, seed, nonce);
  stream256_squeezeblocks(buf, POLY_UNIFORM_ETA2_NBLOCKS, &state);
  unsigned int ctr = rej_eta_mldsa_to(a, PARAM_N, buf,
                                      POLY_UNIFORM_ETA2_NBLOCKS * STREAM256_BLOCKBYTES,
                                      PARAM_ETA_S2);
  while (ctr < PARAM_N) {
    stream256_squeezeblocks(buf, 1, &state);
    ctr += rej_eta_mldsa_to(a + ctr, PARAM_N - ctr, buf,
                            STREAM256_BLOCKBYTES, PARAM_ETA_S2);
  }
#elif ALGORITHM == ALGO_AIGIS
  aigis_shake256_eta_init(&state, seed, (uint8_t)nonce);
  stream256_squeezeblocks(buf, 2, &state);

#if PARAM_ETA_S2 == 3
  rej_eta2(a, PARAM_N, buf);
#elif PARAM_ETA_S2 == 5
  {
    unsigned int pos = rej_eta2(a, 223, buf);

    if (2u * STREAM256_BLOCKBYTES - pos < 85) {
      stream256_squeezeblocks(buf + 2 * STREAM256_BLOCKBYTES, 1, &state);
    }

    rej_eta2(&a[223], 33, &buf[pos]);
  }
#endif
#endif
}

/* ================================================================
 *  gamma1 uniform mask vector y
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA
/* ML-DSA: deterministic unpack from SHAKE stream (GAMMA1-coeff encoding) */
#define POLY_UNIFORM_GAMMA1_NBLOCKS \
  ((POLYZ_PACKEDBYTES + STREAM256_BLOCKBYTES - 1) / STREAM256_BLOCKBYTES)

static __device__ void polyz_unpack(poly *r, const uint8_t *a);  /* forward decl */

static __device__ void poly_uniform_gamma1(poly *a, const uint8_t seed[CRHBYTES], uint16_t nonce) {
  uint8_t buf[POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES];
  stream256_state state;
  stream256_init(&state, seed, nonce);
  stream256_squeezeblocks(buf, POLY_UNIFORM_GAMMA1_NBLOCKS, &state);
  polyz_unpack(a, buf);
}

#elif ALGORITHM == ALGO_AIGIS
/* Aigis: rejection sampling, output Q+GAMMA1-1-t, seed = key||hash (SEEDBYTES+CRHBYTES) */
#define POLY_UNIFORM_GAMMA1_NBLOCKS 5   /* 5 SHAKE256 blocks is conservative */

static __device__ __noinline__ void poly_uniform_gamma1(poly *a,
                                                         const uint8_t seed[SEEDBYTES + CRHBYTES],
                                                         uint16_t nonce) {
  unsigned int ctr = 0, pos = 0;
  uint32_t t0, t1;
  uint8_t buf[POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES];
  stream256_state state;
  aigis_shake256_gamma1_init(&state, seed, nonce);
  stream256_squeezeblocks(buf, POLY_UNIFORM_GAMMA1_NBLOCKS, &state);

  while (ctr < PARAM_N) {
    if (pos + 5 > POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES) {
      /* Squeeze more blocks if needed (very rare) */
      stream256_squeezeblocks(buf, 1, &state);
      pos = 0;
    }
    t0  = buf[pos];
    t0 |= (uint32_t)buf[pos + 1] << 8;
    t0 |= (uint32_t)buf[pos + 2] << 16;
    t1  = buf[pos + 2] >> 4;
    t1 |= (uint32_t)buf[pos + 3] << 4;
    t1 |= (uint32_t)buf[pos + 4] << 12;
    t0 &= 0x3FFFF;
    t1 &= 0x3FFFF;
    pos += 5;
    if (t0 <= 2u * (uint32_t)PARAM_GAMMA1)
      a->coeffs[ctr++] = PARAM_Q + PARAM_GAMMA1 - 1 - (int32_t)t0;
    if (t1 <= 2u * (uint32_t)PARAM_GAMMA1 && ctr < PARAM_N)
      a->coeffs[ctr++] = PARAM_Q + PARAM_GAMMA1 - 1 - (int32_t)t1;
  }
}
#endif

/* ================================================================
 *  Challenge polynomial
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA
/* ML-DSA: absorb CTILDEBYTES seed, range N-TAU..N, coeffs {-1,+1} */
static __device__ __noinline__ void poly_challenge(poly *c, const uint8_t seed[CTILDEBYTES]) {
  unsigned int i, b, pos;
  uint64_t signs;
  uint8_t buf[SHAKE256_RATE];
  keccak_state state;
  shake256_init(&state);
  shake256_absorb(&state, seed, CTILDEBYTES);
  shake256_finalize(&state);
  shake256_squeezeblocks(buf, 1, &state);

  signs = 0;
  for (i = 0; i < 8; ++i) signs |= (uint64_t)buf[i] << 8 * i;
  pos = 8;
  for (i = 0; i < PARAM_N; ++i) c->coeffs[i] = 0;
  for (i = PARAM_N - PARAM_TAU; i < PARAM_N; ++i) {
    do {
      if (pos >= SHAKE256_RATE) { shake256_squeezeblocks(buf, 1, &state); pos = 0; }
      b = buf[pos++];
    } while (b > i);
    c->coeffs[i] = c->coeffs[b];
    c->coeffs[b] = 1 - 2 * (int32_t)(signs & 1);
    signs >>= 1;
  }
}

#elif ALGORITHM == ALGO_AIGIS
/* Aigis: absorb mu(CRHBYTES) + packed_w1, range 196..255 (60 coeffs), coeffs {1, Q-1} */
static __device__ __noinline__ void poly_challenge(poly *c,
                                                    const uint8_t mu[CRHBYTES],
                                                    const uint8_t *packed_w1,
                                                    unsigned int w1_len) {
  unsigned int i, b, pos;
  uint64_t signs, mask;
  uint8_t buf[SHAKE256_RATE];
  keccak_state state;
  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, packed_w1, w1_len);
  shake256_finalize(&state);
  shake256_squeezeblocks(buf, 1, &state);

  signs = 0;
  for (i = 0; i < 8; ++i) signs |= (uint64_t)buf[i] << 8 * i;
  pos = 8;
  mask = 1;
  for (i = 0; i < PARAM_N; ++i) c->coeffs[i] = 0;
  /* Aigis: indices 196..255 (fixed TAU=60 non-zero positions) */
  for (i = 196; i < 256; ++i) {
    do {
      if (pos >= SHAKE256_RATE) { shake256_squeezeblocks(buf, 1, &state); pos = 0; }
      b = buf[pos++];
    } while (b > i);
    c->coeffs[i] = c->coeffs[b];
    c->coeffs[b] = (signs & mask) ? PARAM_Q - 1 : 1;
    mask <<= 1;
  }
}
#endif

/* ================================================================
 *  Packing — 使用 COEFF_BIAS 消除 eta/t0 的算法分叉
 *  ML-DSA (COEFF_BIAS=0): pack as (ETA - coeff)
 *  Aigis  (COEFF_BIAS=Q): pack as (Q + ETA - coeff)
 * ================================================================ */

/* polyeta_s1: SETA1BITS bits per coeff */
static __device__ void polyeta_s1_pack(uint8_t *r, const poly *a) {
  unsigned int i; uint8_t t[8];
#if SETA1BITS == 2
  for (i = 0; i < PARAM_N / 4; ++i) {
    t[0]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[4*i+0]); t[1]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[4*i+1]);
    t[2]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[4*i+2]); t[3]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[4*i+3]);
    r[i] = t[0] | (t[1]<<2) | (t[2]<<4) | (t[3]<<6);
  }
#elif SETA1BITS == 3
  for (i = 0; i < PARAM_N / 8; ++i) {
    t[0]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[8*i+0]); t[1]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[8*i+1]);
    t[2]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[8*i+2]); t[3]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[8*i+3]);
    t[4]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[8*i+4]); t[5]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[8*i+5]);
    t[6]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[8*i+6]); t[7]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S1-a->coeffs[8*i+7]);
    r[3*i+0] = t[0] | (t[1]<<3) | (t[2]<<6);
    r[3*i+1] = (t[2]>>2) | (t[3]<<1) | (t[4]<<4) | (t[5]<<7);
    r[3*i+2] = (t[5]>>1) | (t[6]<<2) | (t[7]<<5);
  }
#elif SETA1BITS == 4
  for (i = 0; i < PARAM_N / 2; ++i) {
    t[0]=(uint8_t)(PARAM_ETA_S1-a->coeffs[2*i+0]); t[1]=(uint8_t)(PARAM_ETA_S1-a->coeffs[2*i+1]);
    r[i] = t[0] | (t[1]<<4);
  }
#endif
}

static __device__ void polyeta_s1_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
#if SETA1BITS == 2
  for (i = 0; i < PARAM_N / 4; ++i) {
    r->coeffs[4*i+0] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)(a[i] & 0x03);
    r->coeffs[4*i+1] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)((a[i]>>2) & 0x03);
    r->coeffs[4*i+2] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)((a[i]>>4) & 0x03);
    r->coeffs[4*i+3] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)((a[i]>>6) & 0x03);
  }
#elif SETA1BITS == 3
  for (i = 0; i < PARAM_N / 8; ++i) {
    r->coeffs[8*i+0] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)(a[3*i+0] & 0x07);
    r->coeffs[8*i+1] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)((a[3*i+0]>>3) & 0x07);
    r->coeffs[8*i+2] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)(((a[3*i+0]>>6)|(a[3*i+1]<<2)) & 0x07);
    r->coeffs[8*i+3] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)((a[3*i+1]>>1) & 0x07);
    r->coeffs[8*i+4] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)((a[3*i+1]>>4) & 0x07);
    r->coeffs[8*i+5] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)(((a[3*i+1]>>7)|(a[3*i+2]<<1)) & 0x07);
    r->coeffs[8*i+6] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)((a[3*i+2]>>2) & 0x07);
    r->coeffs[8*i+7] = COEFF_BIAS + PARAM_ETA_S1 - (int32_t)((a[3*i+2]>>5) & 0x07);
  }
#elif SETA1BITS == 4
  for (i = 0; i < PARAM_N / 2; ++i) {
    r->coeffs[2*i+0] = PARAM_ETA_S1 - (int32_t)(a[i] & 0x0F);
    r->coeffs[2*i+1] = PARAM_ETA_S1 - (int32_t)(a[i] >> 4);
  }
#endif
}

/* polyeta_s2: SETA2BITS bits per coeff */
static __device__ void polyeta_s2_pack(uint8_t *r, const poly *a) {
  unsigned int i; uint8_t t[8];
#if SETA2BITS == 3
  for (i = 0; i < PARAM_N / 8; ++i) {
    t[0]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[8*i+0]); t[1]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[8*i+1]);
    t[2]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[8*i+2]); t[3]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[8*i+3]);
    t[4]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[8*i+4]); t[5]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[8*i+5]);
    t[6]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[8*i+6]); t[7]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[8*i+7]);
    r[3*i+0] = t[0] | (t[1]<<3) | (t[2]<<6);
    r[3*i+1] = (t[2]>>2) | (t[3]<<1) | (t[4]<<4) | (t[5]<<7);
    r[3*i+2] = (t[5]>>1) | (t[6]<<2) | (t[7]<<5);
  }
#elif SETA2BITS == 4
  for (i = 0; i < PARAM_N / 2; ++i) {
    t[0]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[2*i+0]); t[1]=(uint8_t)(COEFF_BIAS+PARAM_ETA_S2-a->coeffs[2*i+1]);
    r[i] = t[0] | (t[1]<<4);
  }
#endif
}

static __device__ void polyeta_s2_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
#if SETA2BITS == 3
  for (i = 0; i < PARAM_N / 8; ++i) {
    r->coeffs[8*i+0] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)(a[3*i+0] & 0x07);
    r->coeffs[8*i+1] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)((a[3*i+0]>>3) & 0x07);
    r->coeffs[8*i+2] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)(((a[3*i+0]>>6)|(a[3*i+1]<<2)) & 0x07);
    r->coeffs[8*i+3] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)((a[3*i+1]>>1) & 0x07);
    r->coeffs[8*i+4] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)((a[3*i+1]>>4) & 0x07);
    r->coeffs[8*i+5] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)(((a[3*i+1]>>7)|(a[3*i+2]<<1)) & 0x07);
    r->coeffs[8*i+6] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)((a[3*i+2]>>2) & 0x07);
    r->coeffs[8*i+7] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)((a[3*i+2]>>5) & 0x07);
  }
#elif SETA2BITS == 4
  for (i = 0; i < PARAM_N / 2; ++i) {
    r->coeffs[2*i+0] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)(a[i] & 0x0F);
    r->coeffs[2*i+1] = COEFF_BIAS + PARAM_ETA_S2 - (int32_t)(a[i] >> 4);
  }
#endif
}

/* polyt1: POLYT1_PACKED_BITS bits per coeff (10 for ML-DSA, 8 for Aigis) */
static __device__ void polyt1_pack(uint8_t *r, const poly *a) {
#if POLYT1_PACKED_BITS == 10
  for (unsigned int i = 0; i < PARAM_N / 4; ++i) {
    r[5*i+0] = (uint8_t)(a->coeffs[4*i+0]);
    r[5*i+1] = (uint8_t)((a->coeffs[4*i+0]>>8) | (a->coeffs[4*i+1]<<2));
    r[5*i+2] = (uint8_t)((a->coeffs[4*i+1]>>6) | (a->coeffs[4*i+2]<<4));
    r[5*i+3] = (uint8_t)((a->coeffs[4*i+2]>>4) | (a->coeffs[4*i+3]<<6));
    r[5*i+4] = (uint8_t)(a->coeffs[4*i+3]>>2);
  }
#elif POLYT1_PACKED_BITS == 8
  for (unsigned int i = 0; i < PARAM_N; ++i) r[i] = (uint8_t)a->coeffs[i];
#endif
}

static __device__ void polyt1_unpack(poly *r, const uint8_t *a) {
#if POLYT1_PACKED_BITS == 10
  for (unsigned int i = 0; i < PARAM_N / 4; ++i) {
    r->coeffs[4*i+0] = ((uint32_t)a[5*i+0] | ((uint32_t)a[5*i+1]<<8)) & 0x3FF;
    r->coeffs[4*i+1] = (((uint32_t)a[5*i+1]>>2) | ((uint32_t)a[5*i+2]<<6)) & 0x3FF;
    r->coeffs[4*i+2] = (((uint32_t)a[5*i+2]>>4) | ((uint32_t)a[5*i+3]<<4)) & 0x3FF;
    r->coeffs[4*i+3] = (((uint32_t)a[5*i+3]>>6) | ((uint32_t)a[5*i+4]<<2)) & 0x3FF;
  }
#elif POLYT1_PACKED_BITS == 8
  for (unsigned int i = 0; i < PARAM_N; ++i) r->coeffs[i] = a[i];
#endif
}

/* polyt0: D bits per coeff, unified with COEFF_BIAS
 * ML-DSA (COEFF_BIAS=0): stored as (2^{D-1} - coeff)
 * Aigis  (COEFF_BIAS=Q): stored as (Q + 2^{D-1} - coeff) */
static __device__ void polyt0_pack(uint8_t *r, const poly *a) {
  unsigned int i; uint32_t t[8];
#if PARAM_D == 13
  for (i = 0; i < PARAM_N / 8; ++i) {
    for (int j = 0; j < 8; j++) t[j] = COEFF_BIAS + (1 << (PARAM_D-1)) - a->coeffs[8*i+j];
    r[13*i+ 0]  =  t[0];        r[13*i+ 1]  =  t[0]>> 8;
    r[13*i+ 1] |=  t[1]<< 5;   r[13*i+ 2]  =  t[1]>> 3;
    r[13*i+ 3]  =  t[1]>>11;   r[13*i+ 3] |=  t[2]<< 2;
    r[13*i+ 4]  =  t[2]>> 6;   r[13*i+ 4] |=  t[3]<< 7;
    r[13*i+ 5]  =  t[3]>> 1;   r[13*i+ 6]  =  t[3]>> 9;
    r[13*i+ 6] |=  t[4]<< 4;   r[13*i+ 7]  =  t[4]>> 4;
    r[13*i+ 8]  =  t[4]>>12;   r[13*i+ 8] |=  t[5]<< 1;
    r[13*i+ 9]  =  t[5]>> 7;   r[13*i+ 9] |=  t[6]<< 6;
    r[13*i+10]  =  t[6]>> 2;   r[13*i+11]  =  t[6]>>10;
    r[13*i+11] |=  t[7]<< 3;   r[13*i+12]  =  t[7]>> 5;
  }
#elif PARAM_D == 14
  for (i = 0; i < PARAM_N / 4; ++i) {
    for (int j = 0; j < 4; j++) t[j] = COEFF_BIAS + (1 << (PARAM_D-1)) - a->coeffs[4*i+j];
    r[7*i+0]  =  t[0];        r[7*i+1]  =  t[0]>> 8;
    r[7*i+1] |=  t[1]<< 6;   r[7*i+2]  =  t[1]>> 2;
    r[7*i+3]  =  t[1]>>10;   r[7*i+3] |=  t[2]<< 4;
    r[7*i+4]  =  t[2]>> 4;   r[7*i+5]  =  t[2]>>12;
    r[7*i+5] |=  t[3]<< 2;   r[7*i+6]  =  t[3]>> 6;
  }
#endif
}

static __device__ void polyt0_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
#if PARAM_D == 13
  for (i = 0; i < PARAM_N / 8; ++i) {
    r->coeffs[8*i+0] = (uint32_t)a[13*i+0] | ((uint32_t)a[13*i+1]<<8); r->coeffs[8*i+0] &= 0x1FFF;
    r->coeffs[8*i+1] = (uint32_t)a[13*i+1]>>5 | ((uint32_t)a[13*i+2]<<3) | ((uint32_t)a[13*i+3]<<11); r->coeffs[8*i+1] &= 0x1FFF;
    r->coeffs[8*i+2] = (uint32_t)a[13*i+3]>>2 | ((uint32_t)a[13*i+4]<<6); r->coeffs[8*i+2] &= 0x1FFF;
    r->coeffs[8*i+3] = (uint32_t)a[13*i+4]>>7 | ((uint32_t)a[13*i+5]<<1) | ((uint32_t)a[13*i+6]<<9); r->coeffs[8*i+3] &= 0x1FFF;
    r->coeffs[8*i+4] = (uint32_t)a[13*i+6]>>4 | ((uint32_t)a[13*i+7]<<4) | ((uint32_t)a[13*i+8]<<12); r->coeffs[8*i+4] &= 0x1FFF;
    r->coeffs[8*i+5] = (uint32_t)a[13*i+8]>>1 | ((uint32_t)a[13*i+9]<<7); r->coeffs[8*i+5] &= 0x1FFF;
    r->coeffs[8*i+6] = (uint32_t)a[13*i+9]>>6 | ((uint32_t)a[13*i+10]<<2) | ((uint32_t)a[13*i+11]<<10); r->coeffs[8*i+6] &= 0x1FFF;
    r->coeffs[8*i+7] = (uint32_t)a[13*i+11]>>3 | ((uint32_t)a[13*i+12]<<5); r->coeffs[8*i+7] &= 0x1FFF;
    for (int j = 0; j < 8; j++) r->coeffs[8*i+j] = COEFF_BIAS + (1 << (PARAM_D-1)) - (int32_t)r->coeffs[8*i+j];
  }
#elif PARAM_D == 14
  for (i = 0; i < PARAM_N / 4; ++i) {
    r->coeffs[4*i+0] = (uint32_t)a[7*i+0] | (((uint32_t)a[7*i+1]&0x3F)<<8); r->coeffs[4*i+0] &= 0x3FFF;
    r->coeffs[4*i+1] = (uint32_t)a[7*i+1]>>6 | ((uint32_t)a[7*i+2]<<2) | (((uint32_t)a[7*i+3]&0x0F)<<10); r->coeffs[4*i+1] &= 0x3FFF;
    r->coeffs[4*i+2] = (uint32_t)a[7*i+3]>>4 | ((uint32_t)a[7*i+4]<<4) | (((uint32_t)a[7*i+5]&0x03)<<12); r->coeffs[4*i+2] &= 0x3FFF;
    r->coeffs[4*i+3] = (uint32_t)a[7*i+5]>>2 | ((uint32_t)a[7*i+6]<<6); r->coeffs[4*i+3] &= 0x3FFF;
    for (int j = 0; j < 4; j++) r->coeffs[4*i+j] = COEFF_BIAS + (1 << (PARAM_D-1)) - (int32_t)r->coeffs[4*i+j];
  }
#endif
}

/* polyz: unified with Z_BIAS/Z_FIXUP
 * ML-DSA (Z_BIAS=GAMMA1):   t = GAMMA1 - coeff
 * Aigis  (Z_BIAS=GAMMA1-1): t = GAMMA1-1 - coeff; 负值+Q */
static __device__ void polyz_pack(uint8_t *r, const poly *a) {
  unsigned int i; uint32_t t[4];
#if PARAM_GAMMA1 == (1 << 17)
  for (i = 0; i < PARAM_N / 4; ++i) {
    t[0]=Z_BIAS-a->coeffs[4*i+0]; Z_FIXUP(t[0]);
    t[1]=Z_BIAS-a->coeffs[4*i+1]; Z_FIXUP(t[1]);
    t[2]=Z_BIAS-a->coeffs[4*i+2]; Z_FIXUP(t[2]);
    t[3]=Z_BIAS-a->coeffs[4*i+3]; Z_FIXUP(t[3]);
    r[9*i+0]=t[0];     r[9*i+1]=t[0]>>8;
    r[9*i+2]=t[0]>>16; r[9*i+2]|=t[1]<<2;
    r[9*i+3]=t[1]>>6;  r[9*i+4]=t[1]>>14;
    r[9*i+4]|=t[2]<<4; r[9*i+5]=t[2]>>4;
    r[9*i+6]=t[2]>>12; r[9*i+6]|=t[3]<<6;
    r[9*i+7]=t[3]>>2;  r[9*i+8]=t[3]>>10;
  }
#elif PARAM_GAMMA1 == (1 << 19)
  for (i = 0; i < PARAM_N / 2; ++i) {
    t[0]=Z_BIAS-a->coeffs[2*i+0]; Z_FIXUP(t[0]);
    t[1]=Z_BIAS-a->coeffs[2*i+1]; Z_FIXUP(t[1]);
    r[5*i+0]=t[0];     r[5*i+1]=t[0]>>8;
    r[5*i+2]=t[0]>>16; r[5*i+2]|=t[1]<<4;
    r[5*i+3]=t[1]>>4;  r[5*i+4]=t[1]>>12;
  }
#endif
}

static __device__ void polyz_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
#if PARAM_GAMMA1 == (1 << 17)
  for (i = 0; i < PARAM_N / 4; ++i) {
    r->coeffs[4*i+0]=((uint32_t)a[9*i+0]|((uint32_t)a[9*i+1]<<8)|((uint32_t)a[9*i+2]<<16))&0x3FFFF;
    r->coeffs[4*i+1]=(((uint32_t)a[9*i+2]>>2)|((uint32_t)a[9*i+3]<<6)|((uint32_t)a[9*i+4]<<14))&0x3FFFF;
    r->coeffs[4*i+2]=(((uint32_t)a[9*i+4]>>4)|((uint32_t)a[9*i+5]<<4)|((uint32_t)a[9*i+6]<<12))&0x3FFFF;
    r->coeffs[4*i+3]=(((uint32_t)a[9*i+6]>>6)|((uint32_t)a[9*i+7]<<2)|((uint32_t)a[9*i+8]<<10))&0x3FFFF;
    r->coeffs[4*i+0]=Z_BIAS-(int32_t)r->coeffs[4*i+0]; Z_FIXUP(r->coeffs[4*i+0]);
    r->coeffs[4*i+1]=Z_BIAS-(int32_t)r->coeffs[4*i+1]; Z_FIXUP(r->coeffs[4*i+1]);
    r->coeffs[4*i+2]=Z_BIAS-(int32_t)r->coeffs[4*i+2]; Z_FIXUP(r->coeffs[4*i+2]);
    r->coeffs[4*i+3]=Z_BIAS-(int32_t)r->coeffs[4*i+3]; Z_FIXUP(r->coeffs[4*i+3]);
  }
#elif PARAM_GAMMA1 == (1 << 19)
  for (i = 0; i < PARAM_N / 2; ++i) {
    r->coeffs[2*i+0]=((uint32_t)a[5*i+0]|((uint32_t)a[5*i+1]<<8)|((uint32_t)a[5*i+2]<<16))&0xFFFFF;
    r->coeffs[2*i+1]=(((uint32_t)a[5*i+2]>>4)|((uint32_t)a[5*i+3]<<4)|((uint32_t)a[5*i+4]<<12))&0xFFFFF;
    r->coeffs[2*i+0]=Z_BIAS-(int32_t)r->coeffs[2*i+0]; Z_FIXUP(r->coeffs[2*i+0]);
    r->coeffs[2*i+1]=Z_BIAS-(int32_t)r->coeffs[2*i+1]; Z_FIXUP(r->coeffs[2*i+1]);
  }
#endif
}

/* polyw1: encode coeff in [0, N_W1) */
static __device__ void polyw1_pack(uint8_t *r, const poly *a) {
  unsigned int i;
#if PARAM_GAMMA2 == (PARAM_Q - 1) / 88     /* N_W1=44: 6 bits, 4 per 3 bytes */
  for (i = 0; i < PARAM_N / 4; ++i) {
    r[3*i+0]  = a->coeffs[4*i+0] | (a->coeffs[4*i+1]<<6);
    r[3*i+1]  = (a->coeffs[4*i+1]>>2) | (a->coeffs[4*i+2]<<4);
    r[3*i+2]  = (a->coeffs[4*i+2]>>4) | (a->coeffs[4*i+3]<<2);
  }
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 32   /* N_W1=16: 4 bits (nibble), 2 per byte */
  for (i = 0; i < PARAM_N / 2; ++i)
    r[i] = (uint8_t)(a->coeffs[2*i+0] | (a->coeffs[2*i+1]<<4));
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 12   /* N_W1=6: 3 bits, 8 per 3 bytes */
  for (i = 0; i < PARAM_N / 8; ++i) {
    r[3*i+0] = a->coeffs[8*i+0] | (a->coeffs[8*i+1]<<3) | (a->coeffs[8*i+2]<<6);
    r[3*i+1] = (a->coeffs[8*i+2]>>2) | (a->coeffs[8*i+3]<<1) | (a->coeffs[8*i+4]<<4) | (a->coeffs[8*i+5]<<7);
    r[3*i+2] = (a->coeffs[8*i+5]>>1) | (a->coeffs[8*i+6]<<2) | (a->coeffs[8*i+7]<<5);
  }
#endif
}

#endif /* POLY_CUH */
