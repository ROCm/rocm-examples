#ifndef SIGN_CUH
#define SIGN_CUH

#include <stdint.h>
#include <string.h>
#include "params.h"
#include "packing.cuh"
#include "polyvec.cuh"
#include "poly.cuh"
#include "symmetric.cuh"
#include "fips202.cuh"

/* ================================================================
 *  KEY GENERATION (unified skeleton, 4 inner #if blocks)
 * ================================================================ */
static __device__ __noinline__ int crypto_sign_keypair(
    uint8_t *pk, uint8_t *sk, const uint8_t *seed)
{
  polyvecl mat[PARAM_K];
  polyvecl s1, s1hat;
  polyveck s2, t1, t0;

  /* ---- seed derivation (algo-specific) ---- */
#if ALGORITHM == ALGO_MLDSA
  uint8_t seedbuf[2 * SEEDBYTES + CRHBYTES];
  uint8_t tr[TRBYTES];
  memcpy(seedbuf, seed, SEEDBYTES);
  seedbuf[SEEDBYTES]     = (uint8_t)PARAM_K;
  seedbuf[SEEDBYTES + 1] = (uint8_t)PARAM_L;
  shake256(seedbuf, 2 * SEEDBYTES + CRHBYTES, seedbuf, SEEDBYTES + 2);
  const uint8_t *rho      = seedbuf;
  const uint8_t *eta_seed = seedbuf + SEEDBYTES;           /* rhoprime */
  const uint8_t *key      = seedbuf + SEEDBYTES + CRHBYTES;
#elif ALGORITHM == ALGO_AIGIS
  uint8_t buf[3 * SEEDBYTES + CRHBYTES];
  shake256(buf, 3 * SEEDBYTES, seed, SEEDBYTES);
  const uint8_t *eta_seed = buf;                            /* sampling_seed */
  const uint8_t *rho      = buf + SEEDBYTES;
  const uint8_t *key      = buf + 2 * SEEDBYTES;
#endif

  /* ---- shared: expand A, sample s1/s2 ---- */
  polyvec_matrix_expand(mat, rho);
  polyvecl_uniform_eta_s1(&s1, eta_seed, 0);
  polyveck_uniform_eta_s2(&s2, eta_seed, PARAM_L);

  /* ---- shared: t = A*NTT(s1) + s2, then power2round ---- */
  s1hat = s1;
  polyvecl_ntt(&s1hat);
  polyveck_accumulate_matvecntt(&t1, mat, &s1hat);
#if ALGORITHM == ALGO_MLDSA
  polyveck_reduce(&t1);
#endif
  polyveck_invntt_tomont(&t1);
  polyveck_add(&t1, &t1, &s2);
#if ALGORITHM == ALGO_MLDSA
  polyveck_caddq(&t1);
#elif ALGORITHM == ALGO_AIGIS
  polyveck_freeze4q(&t1);
#endif
  polyveck_power2round(&t1, &t0, &t1);
  pack_pk(pk, rho, &t1);

  /* ---- hash pk and pack sk (algo-specific) ---- */
#if ALGORITHM == ALGO_MLDSA
  shake256(tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  pack_sk(sk, rho, key, tr, &s1, &s2, &t0);
#elif ALGORITHM == ALGO_AIGIS
  shake256(buf + 3 * SEEDBYTES, CRHBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  pack_sk(sk, rho, key, buf + 3 * SEEDBYTES, &s1, &s2, &t0);
#endif

  return 0;
}

/* ================================================================
 *  SIGNATURE
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __noinline__ int crypto_sign_signature(
    uint8_t *sig, size_t *siglen,
    const uint8_t *m, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *rnd_in,
    const uint8_t *sk)
{
  unsigned int n;
  uint8_t seedbuf[2 * SEEDBYTES + TRBYTES + 2 * CRHBYTES];
  uint8_t *rho, *tr, *key, *mu, *rhoprime;
  uint16_t nonce = 0;
  polyvecl mat[PARAM_K], s1, y, z;
  polyveck t0, s2, w1, w0, h;
  poly cp;
  keccak_state state;

  rho      = seedbuf;
  tr       = rho + SEEDBYTES;
  key      = tr + TRBYTES;
  mu       = key + SEEDBYTES;
  rhoprime = mu + CRHBYTES;
  unpack_sk(rho, key, tr, &s1, &s2, &t0, sk);

  shake256_init(&state);
  shake256_absorb(&state, tr, TRBYTES);
  shake256_absorb(&state, pre, prelen);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  shake256_init(&state);
  shake256_absorb(&state, key, SEEDBYTES);
#if RNDBYTES > 0
  shake256_absorb(&state, rnd_in, RNDBYTES);
#endif
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_finalize(&state);
  shake256_squeeze(rhoprime, CRHBYTES, &state);

  polyvec_matrix_expand(mat, rho);
  polyvecl_ntt(&s1);
  polyveck_ntt(&s2);
  polyveck_ntt(&t0);

rej:
  polyvecl_uniform_gamma1(&y, rhoprime, nonce++);

  z = y;
  polyvecl_ntt(&z);
  polyveck_accumulate_matvecntt(&w1, mat, &z);
  polyveck_reduce(&w1);
  polyveck_invntt_tomont(&w1);

  polyveck_reduce(&w1);
  polyveck_caddq(&w1);
  polyveck_decompose(&w1, &w0, &w1);
  polyveck_pack_w1(sig, &w1);

  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, sig, PARAM_K * POLYW1_PACKEDBYTES);
  shake256_finalize(&state);
  shake256_squeeze(sig, CTILDEBYTES, &state);
  poly_challenge(&cp, sig);
  poly_ntt(&cp);

  polyvecl_pointwise_poly_montgomery(&z, &cp, &s1);
  polyvecl_invntt_tomont(&z);
  polyvecl_add(&z, &z, &y);
  polyvecl_reduce(&z);
  if (polyvecl_chknorm(&z, PARAM_GAMMA1 - PARAM_BETA1))
    goto rej;

  polyveck_pointwise_poly_montgomery(&h, &cp, &s2);
  polyveck_invntt_tomont(&h);
  polyveck_sub(&w0, &w0, &h);
  polyveck_reduce(&w0);
  if (polyveck_chknorm(&w0, PARAM_GAMMA2 - PARAM_BETA2))
    goto rej;

  polyveck_pointwise_poly_montgomery(&h, &cp, &t0);
  polyveck_invntt_tomont(&h);
  polyveck_reduce(&h);
  if (polyveck_chknorm(&h, PARAM_GAMMA2))
    goto rej;

  polyveck_add(&w0, &w0, &h);
  n = polyveck_make_hint(&h, &w0, &w1);
  if (n > PARAM_OMEGA)
    goto rej;

  pack_sig(sig, sig, &z, &h);
  *siglen = CRYPTO_BYTES;
  return 0;
}

#elif ALGORITHM == ALGO_AIGIS

static __device__ __noinline__ int crypto_sign_signature(
    uint8_t *sig, size_t *siglen,
    const uint8_t *m, size_t mlen,
    const uint8_t *rnd_in,
    const uint8_t *sk)
{
  unsigned int n;
  uint8_t rho[SEEDBYTES], key[SEEDBYTES], hash_pk[TRBYTES];
  uint8_t mu[CRHBYTES];
  uint8_t key_mu[SEEDBYTES + CRHBYTES];  /* gamma1 seed = key || mu */
  uint8_t w1_buf[PARAM_K * POLYW1_PACKEDBYTES];
  uint16_t nonce = 0;
  polyvecl mat[PARAM_K], s1, y, z;
  polyveck t0, s2, w, w1, wcs2, wcs20, ct0, h, tmp;
  poly c, chat;
  keccak_state state;

  unpack_sk(rho, key, hash_pk, &s1, &s2, &t0, sk);

  /* mu = shake256(hash_pk || m) */
  shake256_init(&state);
  shake256_absorb(&state, hash_pk, TRBYTES);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  /* gamma1 seed = key || mu */
  memcpy(key_mu, key, SEEDBYTES);
  memcpy(key_mu + SEEDBYTES, mu, CRHBYTES);

  polyvec_matrix_expand(mat, rho);
  polyvecl_ntt(&s1);
  polyveck_ntt(&s2);
  polyveck_ntt(&t0);

rej:

  polyvecl_uniform_gamma1(&y, key_mu, nonce);
  nonce += PARAM_L;

  z = y;
  polyvecl_ntt(&z);
  polyveck_accumulate_matvecntt(&w, mat, &z); /* barrat_reduce included */
  polyveck_invntt_tomont(&w);

  polyveck_freeze2q(&w);
  polyveck_decompose(&w1, &tmp, &w);

  /* Aigis: challenge from mu || packed_w1 */
  polyveck_pack_w1(w1_buf, &w1);
  poly_challenge(&c, mu, w1_buf, PARAM_K * POLYW1_PACKEDBYTES);

  chat = c;
  poly_ntt(&chat);

  /* z = chat*s1 + y */
  polyvecl_pointwise_poly_montgomery(&z, &chat, &s1);
  polyvecl_invntt_tomont(&z);
  polyvecl_add(&z, &z, &y);
  polyvecl_freeze4q(&z);
  if (polyvecl_chknorm(&z, PARAM_GAMMA1 - PARAM_BETA1))
    goto rej;

  /* wcs2 = w - chat*s2; decompose; check high bits == w1 */
  polyveck_pointwise_poly_montgomery(&wcs2, &chat, &s2);
  polyveck_invntt_tomont(&wcs2);
  polyveck_sub(&wcs2, &w, &wcs2);
  polyveck_freeze4q(&wcs2);
  polyveck_decompose(&tmp, &wcs20, &wcs2);
  polyveck_freeze2q(&wcs20);
  if (polyveck_chknorm(&wcs20, PARAM_GAMMA2 - PARAM_BETA2))
    goto rej;

  {
    int _w1_mismatch = 0;
    for (unsigned int i = 0; i < PARAM_K && !_w1_mismatch; ++i)
      for (unsigned int j = 0; j < PARAM_N && !_w1_mismatch; ++j)
        if (tmp.vec[i].coeffs[j] != w1.vec[i].coeffs[j])
          _w1_mismatch = 1;
    if (_w1_mismatch)
      goto rej;
  }

  /* ct0 = chat*t0 */
  polyveck_pointwise_poly_montgomery(&ct0, &chat, &t0);
  polyveck_invntt_tomont(&ct0);
  polyveck_freeze2q(&ct0);
  if (polyveck_chknorm(&ct0, PARAM_GAMMA2))
    goto rej;

  /* make_hint: h = hint(wcs2+ct0, neg(ct0)) */
  polyveck_add(&tmp, &wcs2, &ct0);
  polyveck_neg(&ct0);
  polyveck_freeze2q(&tmp);
  n = polyveck_make_hint(&h, &tmp, &ct0);
  if (n > PARAM_OMEGA)
    goto rej;

  pack_sig(sig, &z, &h, &c);
  *siglen = CRYPTO_BYTES;
  return 0;
}

#endif /* ALGORITHM sign */

/* ================================================================
 *  VERIFY
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __noinline__ int crypto_sign_verify(
    const uint8_t *sig, size_t siglen,
    const uint8_t *m, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *pk)
{
  unsigned int i;
  uint8_t buf[PARAM_K * POLYW1_PACKEDBYTES];
  uint8_t rho[SEEDBYTES];
  uint8_t mu[CRHBYTES];
  uint8_t c[CTILDEBYTES];
  uint8_t c2[CTILDEBYTES];
  poly cp;
  polyvecl mat[PARAM_K], z;
  polyveck t1, w1, h;
  keccak_state state;

  if (siglen != CRYPTO_BYTES)
    return -1;

  unpack_pk(rho, &t1, pk);
  if (unpack_sig(c, &z, &h, sig))
    return -1;
  if (polyvecl_chknorm(&z, PARAM_GAMMA1 - PARAM_BETA1))
    return -1;

  shake256(mu, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  shake256_init(&state);
  shake256_absorb(&state, mu, TRBYTES);
  shake256_absorb(&state, pre, prelen);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  poly_challenge(&cp, c);
  polyvec_matrix_expand(mat, rho);

  polyvecl_ntt(&z);
  polyveck_accumulate_matvecntt(&w1, mat, &z);

  poly_ntt(&cp);
  polyveck_shiftl(&t1);
  polyveck_ntt(&t1);
  polyveck_pointwise_poly_montgomery(&t1, &cp, &t1);

  polyveck_sub(&w1, &w1, &t1);
  polyveck_reduce(&w1);
  polyveck_invntt_tomont(&w1);

  polyveck_reduce(&w1);
  polyveck_caddq(&w1);
  polyveck_use_hint(&w1, &w1, &h);
  polyveck_pack_w1(buf, &w1);

  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, buf, PARAM_K * POLYW1_PACKEDBYTES);
  shake256_finalize(&state);
  shake256_squeeze(c2, CTILDEBYTES, &state);
  for (i = 0; i < CTILDEBYTES; ++i)
    if (c[i] != c2[i])
      return -1;

  return 0;
}

#elif ALGORITHM == ALGO_AIGIS

static __device__ __noinline__ int crypto_sign_verify(
    const uint8_t *sig, size_t siglen,
    const uint8_t *m, size_t mlen,
    const uint8_t *pk)
{
  uint8_t rho[SEEDBYTES];
  uint8_t mu[CRHBYTES];
  uint8_t w1_buf[PARAM_K * POLYW1_PACKEDBYTES];
  poly c, cp, chat;
  polyvecl mat[PARAM_K], z;
  polyveck t1, w1, h, tmp1, tmp2;
  keccak_state state;

  if (siglen != CRYPTO_BYTES)
    return -1;

  unpack_pk(rho, &t1, pk);
  if (unpack_sig(&z, &h, &c, sig))
    return -1;
  if (polyvecl_chknorm(&z, PARAM_GAMMA1 - PARAM_BETA1))
    return -1;

  /* mu = shake256(shake256(pk) || m) */
  shake256(mu, CRHBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  polyvec_matrix_expand(mat, rho);

  polyvecl_ntt(&z);
  polyveck_accumulate_matvecntt(&tmp1, mat, &z); /* barrat_reduce included */

  chat = c;
  poly_ntt(&chat);
  polyveck_shiftl(&t1);
  polyveck_ntt(&t1);
  polyveck_pointwise_poly_montgomery(&tmp2, &chat, &t1);

  polyveck_sub(&tmp1, &tmp1, &tmp2);
  polyveck_reduce(&tmp1);  /* Remove 2*Q bias from poly_sub before INVNTT */
  polyveck_invntt_tomont(&tmp1);

  polyveck_freeze2q(&tmp1);
  polyveck_use_hint(&w1, &tmp1, &h);

  /* Recompute challenge and compare coefficients */
  polyveck_pack_w1(w1_buf, &w1);
  poly_challenge(&cp, mu, w1_buf, PARAM_K * POLYW1_PACKEDBYTES);
  for (unsigned int i = 0; i < PARAM_N; ++i)
    if (c.coeffs[i] != cp.coeffs[i])
      return -1;

  return 0;
}

#endif /* ALGORITHM verify */

/* ================================================================
 *  PRECOMPUTATION — 预计算结构和函数
 *
 *  用于同一密钥的批量签名/验证场景。
 *  预计算内容:
 *    mat[K][L] — 扩展后的矩阵 A (NTT 域)
 *    s1_ntt    — NTT(s1) (仅签名)
 *    s2_ntt    — NTT(s2) (仅签名)
 *    t0_ntt    — NTT(t0) (仅签名)
 *    key, tr   — 种子材料
 * ================================================================ */
typedef struct {
    polyvecl mat[PARAM_K];     /* 扩展矩阵 A (NTT 域) */
    polyvecl s1_ntt;           /* NTT(s1) — 签名用 */
    polyveck s2_ntt;           /* NTT(s2) — 签名用 */
    polyveck t0_ntt;           /* NTT(t0) — 签名用 */
    uint8_t  key[SEEDBYTES];   /* 签名用: rhoprime 推导 */
    uint8_t  tr[TRBYTES];     /* 签名/验证用: mu 计算 */
} precomp_t;

/* 创建预计算数据: 从 pk/sk 提取并预计算 */
static __device__ __noinline__ void create_precomp(
    precomp_t *pc,
    const uint8_t *pk,
    const uint8_t *sk)
{
    uint8_t rho[SEEDBYTES];
    /* unpack_sk 直接写入 pc 的存储, 避免额外拷贝 */
    unpack_sk(rho, pc->key, pc->tr, &pc->s1_ntt, &pc->s2_ntt, &pc->t0_ntt, sk);
    polyvec_matrix_expand(pc->mat, rho);
    polyvecl_ntt(&pc->s1_ntt);
    polyveck_ntt(&pc->s2_ntt);
    polyveck_ntt(&pc->t0_ntt);
}

/* ================================================================
 *  预计算签名 — 跳过矩阵扩展和密钥 NTT
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __noinline__ int crypto_sign_signature_precomp_cached(
    uint8_t *sig, size_t *siglen,
    const uint8_t mu[CRHBYTES],
    const uint8_t rhoprime[CRHBYTES],
    const precomp_t *pc,
    uint16_t nonce_start)
{
  unsigned int n;
  uint16_t nonce = nonce_start;
  polyvecl y, z;
  polyveck w1, w0, h;
  poly cp;
  keccak_state state;

rej_p_cached:
  polyvecl_uniform_gamma1(&y, rhoprime, nonce++);

  z = y;
  polyvecl_ntt(&z);
  polyveck_accumulate_matvecntt(&w1, pc->mat, &z);
  polyveck_reduce(&w1);
  polyveck_invntt_tomont(&w1);
  polyveck_reduce(&w1);
  polyveck_caddq(&w1);
  polyveck_decompose(&w1, &w0, &w1);
  polyveck_pack_w1(sig, &w1);

  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, sig, PARAM_K * POLYW1_PACKEDBYTES);
  shake256_finalize(&state);
  shake256_squeeze(sig, CTILDEBYTES, &state);
  poly_challenge(&cp, sig);
  poly_ntt(&cp);

  polyvecl_pointwise_poly_montgomery(&z, &cp, &pc->s1_ntt);
  polyvecl_invntt_tomont(&z);
  polyvecl_add(&z, &z, &y);
  polyvecl_reduce(&z);
  if (polyvecl_chknorm(&z, PARAM_GAMMA1 - PARAM_BETA1))
    goto rej_p_cached;

  polyveck_pointwise_poly_montgomery(&h, &cp, &pc->s2_ntt);
  polyveck_invntt_tomont(&h);
  polyveck_sub(&w0, &w0, &h);
  polyveck_reduce(&w0);
  if (polyveck_chknorm(&w0, PARAM_GAMMA2 - PARAM_BETA2))
    goto rej_p_cached;

  polyveck_pointwise_poly_montgomery(&h, &cp, &pc->t0_ntt);
  polyveck_invntt_tomont(&h);
  polyveck_reduce(&h);
  if (polyveck_chknorm(&h, PARAM_GAMMA2))
    goto rej_p_cached;

  polyveck_add(&w0, &w0, &h);
  n = polyveck_make_hint(&h, &w0, &w1);
  if (n > PARAM_OMEGA)
    goto rej_p_cached;

  pack_sig(sig, sig, &z, &h);
  *siglen = CRYPTO_BYTES;
  return 0;
}

static __device__ __noinline__ int crypto_sign_signature_precomp(
    uint8_t *sig, size_t *siglen,
    const uint8_t *m, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *rnd_in,
    const precomp_t *pc,
    uint16_t nonce_start)
{
  uint8_t mu[CRHBYTES], rhoprime[CRHBYTES];
  keccak_state state;

  /* mu = H(tr || pre || m) */
  shake256_init(&state);
  shake256_absorb(&state, pc->tr, TRBYTES);
  shake256_absorb(&state, pre, prelen);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  /* rhoprime = H(key || rnd || mu) */
  shake256_init(&state);
  shake256_absorb(&state, pc->key, SEEDBYTES);
#if RNDBYTES > 0
  shake256_absorb(&state, rnd_in, RNDBYTES);
#endif
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_finalize(&state);
  shake256_squeeze(rhoprime, CRHBYTES, &state);

  return crypto_sign_signature_precomp_cached(sig, siglen, mu, rhoprime, pc, nonce_start);
}

#elif ALGORITHM == ALGO_AIGIS

static __device__ __noinline__ int crypto_sign_signature_precomp_cached(
    uint8_t *sig, size_t *siglen,
    const uint8_t mu[CRHBYTES],
    const uint8_t key_mu[SEEDBYTES + CRHBYTES],
    const precomp_t *pc,
    uint16_t nonce_start)
{
  unsigned int n;
  uint8_t w1_buf[PARAM_K * POLYW1_PACKEDBYTES];
  uint16_t nonce = nonce_start;
  polyvecl y, z;
  polyveck w, w1, wcs2, wcs20, ct0, h, tmp;
  poly c, chat;

rej_p_cached:
  polyvecl_uniform_gamma1(&y, key_mu, nonce);
  nonce += PARAM_L;

  z = y;
  polyvecl_ntt(&z);
  polyveck_accumulate_matvecntt(&w, pc->mat, &z);
  polyveck_invntt_tomont(&w);
  polyveck_freeze2q(&w);
  polyveck_decompose(&w1, &tmp, &w);

  polyveck_pack_w1(w1_buf, &w1);
  poly_challenge(&c, mu, w1_buf, PARAM_K * POLYW1_PACKEDBYTES);

  chat = c;
  poly_ntt(&chat);

  polyvecl_pointwise_poly_montgomery(&z, &chat, &pc->s1_ntt);
  polyvecl_invntt_tomont(&z);
  polyvecl_add(&z, &z, &y);
  polyvecl_freeze4q(&z);
  if (polyvecl_chknorm(&z, PARAM_GAMMA1 - PARAM_BETA1))
    goto rej_p_cached;

  polyveck_pointwise_poly_montgomery(&wcs2, &chat, &pc->s2_ntt);
  polyveck_invntt_tomont(&wcs2);
  polyveck_sub(&wcs2, &w, &wcs2);
  polyveck_freeze4q(&wcs2);
  polyveck_decompose(&tmp, &wcs20, &wcs2);
  polyveck_freeze2q(&wcs20);
  if (polyveck_chknorm(&wcs20, PARAM_GAMMA2 - PARAM_BETA2))
    goto rej_p_cached;

  {
    int _w1_mismatch = 0;
    for (unsigned int i = 0; i < PARAM_K && !_w1_mismatch; ++i)
      for (unsigned int j = 0; j < PARAM_N && !_w1_mismatch; ++j)
        if (tmp.vec[i].coeffs[j] != w1.vec[i].coeffs[j])
          _w1_mismatch = 1;
    if (_w1_mismatch)
      goto rej_p_cached;
  }

  polyveck_pointwise_poly_montgomery(&ct0, &chat, &pc->t0_ntt);
  polyveck_invntt_tomont(&ct0);
  polyveck_freeze2q(&ct0);
  if (polyveck_chknorm(&ct0, PARAM_GAMMA2))
    goto rej_p_cached;

  polyveck_add(&tmp, &wcs2, &ct0);
  polyveck_neg(&ct0);
  polyveck_freeze2q(&tmp);
  n = polyveck_make_hint(&h, &tmp, &ct0);
  if (n > PARAM_OMEGA)
    goto rej_p_cached;

  pack_sig(sig, &z, &h, &c);
  *siglen = CRYPTO_BYTES;
  return 0;
}

static __device__ __noinline__ int crypto_sign_signature_precomp(
    uint8_t *sig, size_t *siglen,
    const uint8_t *m, size_t mlen,
    const uint8_t *rnd_in,
    const precomp_t *pc,
    uint16_t nonce_start)
{
  uint8_t mu[CRHBYTES];
  uint8_t key_mu[SEEDBYTES + CRHBYTES];
  keccak_state state;

  /* mu = H(hash_pk || m) */
  shake256_init(&state);
  shake256_absorb(&state, pc->tr, TRBYTES);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  /* gamma1 seed = key || mu */
  memcpy(key_mu, pc->key, SEEDBYTES);
  memcpy(key_mu + SEEDBYTES, mu, CRHBYTES);

  return crypto_sign_signature_precomp_cached(sig, siglen, mu, key_mu, pc, nonce_start);
}

#endif /* ALGORITHM sign_precomp */

/* ================================================================
 *  预计算验证 — 跳过矩阵扩展
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

static __device__ __noinline__ int crypto_sign_verify_precomp(
    const uint8_t *sig, size_t siglen,
    const uint8_t *m, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *pk,
    const polyvecl *precomp_mat)
{
  unsigned int i;
  uint8_t buf[PARAM_K * POLYW1_PACKEDBYTES];
  uint8_t mu[CRHBYTES];
  uint8_t c[CTILDEBYTES];
  uint8_t c2[CTILDEBYTES];
  poly cp;
  polyvecl z;
  polyveck t1, w1, h;
  keccak_state state;

  if (siglen != CRYPTO_BYTES)
    return -1;

  unpack_pk(mu, &t1, pk);  /* mu 暂存 rho, 后面覆盖 */
  if (unpack_sig(c, &z, &h, sig))
    return -1;
  if (polyvecl_chknorm(&z, PARAM_GAMMA1 - PARAM_BETA1))
    return -1;

  shake256(mu, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  shake256_init(&state);
  shake256_absorb(&state, mu, TRBYTES);
  shake256_absorb(&state, pre, prelen);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  poly_challenge(&cp, c);
  /* 跳过 polyvec_matrix_expand — 使用预计算矩阵 */

  polyvecl_ntt(&z);
  polyveck_accumulate_matvecntt(&w1, precomp_mat, &z);

  poly_ntt(&cp);
  polyveck_shiftl(&t1);
  polyveck_ntt(&t1);
  polyveck_pointwise_poly_montgomery(&t1, &cp, &t1);

  polyveck_sub(&w1, &w1, &t1);
  polyveck_reduce(&w1);
  polyveck_invntt_tomont(&w1);

  polyveck_reduce(&w1);
  polyveck_caddq(&w1);
  polyveck_use_hint(&w1, &w1, &h);
  polyveck_pack_w1(buf, &w1);

  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, buf, PARAM_K * POLYW1_PACKEDBYTES);
  shake256_finalize(&state);
  shake256_squeeze(c2, CTILDEBYTES, &state);
  for (i = 0; i < CTILDEBYTES; ++i)
    if (c[i] != c2[i])
      return -1;

  return 0;
}

#elif ALGORITHM == ALGO_AIGIS

static __device__ __noinline__ int crypto_sign_verify_precomp(
    const uint8_t *sig, size_t siglen,
    const uint8_t *m, size_t mlen,
    const uint8_t *pk,
    const polyvecl *precomp_mat)
{
  uint8_t rho[SEEDBYTES];
  uint8_t mu[CRHBYTES];
  uint8_t w1_buf[PARAM_K * POLYW1_PACKEDBYTES];
  poly c, cp, chat;
  polyvecl z;
  polyveck t1, w1, h, tmp1, tmp2;
  keccak_state state;

  if (siglen != CRYPTO_BYTES)
    return -1;

  unpack_pk(rho, &t1, pk);
  if (unpack_sig(&z, &h, &c, sig))
    return -1;
  if (polyvecl_chknorm(&z, PARAM_GAMMA1 - PARAM_BETA1))
    return -1;

  shake256(mu, CRHBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  /* 跳过 polyvec_matrix_expand — 使用预计算矩阵 */

  polyvecl_ntt(&z);
  polyveck_accumulate_matvecntt(&tmp1, precomp_mat, &z);

  chat = c;
  poly_ntt(&chat);
  polyveck_shiftl(&t1);
  polyveck_ntt(&t1);
  polyveck_pointwise_poly_montgomery(&tmp2, &chat, &t1);

  polyveck_sub(&tmp1, &tmp1, &tmp2);
  polyveck_reduce(&tmp1);
  polyveck_invntt_tomont(&tmp1);

  polyveck_freeze2q(&tmp1);
  polyveck_use_hint(&w1, &tmp1, &h);

  polyveck_pack_w1(w1_buf, &w1);
  poly_challenge(&cp, mu, w1_buf, PARAM_K * POLYW1_PACKEDBYTES);
  for (unsigned int i = 0; i < PARAM_N; ++i)
    if (c.coeffs[i] != cp.coeffs[i])
      return -1;

  return 0;
}

#endif /* ALGORITHM verify_precomp */

#endif /* SIGN_CUH */
