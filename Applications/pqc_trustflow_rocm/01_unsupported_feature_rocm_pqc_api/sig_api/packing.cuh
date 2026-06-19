#ifndef PACKING_CUH
#define PACKING_CUH

#include <stdint.h>
#include "params.h"
#include "poly.cuh"
#include "polyvec.cuh"

/*
 * Signature format:
 *   ML-DSA: c_tilde (CTILDEBYTES) || z_packed (L*POLYZ_PACKEDBYTES) || hint_bitmap (OMEGA+K)
 *   Aigis:  z_packed (L*POLYZ_PACKEDBYTES) || hint_bitmap (OMEGA+K) || challenge_poly (N/8+8)
 *
 * Hint bitmap layout: first OMEGA bytes are sorted coeff indices with hints=1,
 *                     last K bytes are end offsets for each poly (same for both).
 */

/* ================================================================
 *  Public key: rho (SEEDBYTES=32) || t1 packed (K * POLYT1_PACKEDBYTES)
 * ================================================================ */
static __device__ void pack_pk(uint8_t pk[CRYPTO_PUBLICKEYBYTES],
                                const uint8_t rho[SEEDBYTES], const polyveck *t1) {
  for (unsigned int i = 0; i < SEEDBYTES; ++i) pk[i] = rho[i];
  for (unsigned int i = 0; i < PARAM_K; ++i)
    polyt1_pack(pk + SEEDBYTES + i * POLYT1_PACKEDBYTES, &t1->vec[i]);
}

static __device__ void unpack_pk(uint8_t rho[SEEDBYTES], polyveck *t1,
                                  const uint8_t pk[CRYPTO_PUBLICKEYBYTES]) {
  for (unsigned int i = 0; i < SEEDBYTES; ++i) rho[i] = pk[i];
  for (unsigned int i = 0; i < PARAM_K; ++i)
    polyt1_unpack(&t1->vec[i], pk + SEEDBYTES + i * POLYT1_PACKEDBYTES);
}

/* ================================================================
 *  Secret key: rho (SEEDBYTES) || key (SEEDBYTES) || tr (TRBYTES)
 *              || s1 (L*POLYETA1_PACKEDBYTES) || s2 (K*POLYETA2_PACKEDBYTES)
 *              || t0 (K*POLYT0_PACKEDBYTES)
 * ================================================================ */
static __device__ void pack_sk(uint8_t sk[CRYPTO_SECRETKEYBYTES],
                                const uint8_t rho[SEEDBYTES],
                                const uint8_t key[SEEDBYTES],
                                const uint8_t tr[TRBYTES],
                                const polyvecl *s1, const polyveck *s2,
                                const polyveck *t0) {
  unsigned int offset = 0;
  for (unsigned int i = 0; i < SEEDBYTES; ++i) sk[offset++] = rho[i];
  for (unsigned int i = 0; i < SEEDBYTES; ++i) sk[offset++] = key[i];
  for (unsigned int i = 0; i < TRBYTES; ++i)   sk[offset++] = tr[i];
  for (unsigned int i = 0; i < PARAM_L; ++i) {
    polyeta_s1_pack(sk + offset, &s1->vec[i]);
    offset += POLYETA1_PACKEDBYTES;
  }
  for (unsigned int i = 0; i < PARAM_K; ++i) {
    polyeta_s2_pack(sk + offset, &s2->vec[i]);
    offset += POLYETA2_PACKEDBYTES;
  }
  for (unsigned int i = 0; i < PARAM_K; ++i) {
    polyt0_pack(sk + offset, &t0->vec[i]);
    offset += POLYT0_PACKEDBYTES;
  }
}

static __device__ void unpack_sk(uint8_t rho[SEEDBYTES], uint8_t key[SEEDBYTES],
                                  uint8_t tr[TRBYTES],
                                  polyvecl *s1, polyveck *s2, polyveck *t0,
                                  const uint8_t sk[CRYPTO_SECRETKEYBYTES]) {
  unsigned int offset = 0;
  for (unsigned int i = 0; i < SEEDBYTES; ++i) rho[i] = sk[offset++];
  for (unsigned int i = 0; i < SEEDBYTES; ++i) key[i] = sk[offset++];
  for (unsigned int i = 0; i < TRBYTES; ++i)   tr[i]  = sk[offset++];
  for (unsigned int i = 0; i < PARAM_L; ++i) {
    polyeta_s1_unpack(&s1->vec[i], sk + offset);
    offset += POLYETA1_PACKEDBYTES;
  }
  for (unsigned int i = 0; i < PARAM_K; ++i) {
    polyeta_s2_unpack(&s2->vec[i], sk + offset);
    offset += POLYETA2_PACKEDBYTES;
  }
  for (unsigned int i = 0; i < PARAM_K; ++i) {
    polyt0_unpack(&t0->vec[i], sk + offset);
    offset += POLYT0_PACKEDBYTES;
  }
}

/* ================================================================
 *  Signature packing/unpacking — algorithm-specific format
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA
/* ML-DSA: c_tilde || z || hint_bitmap */
static __device__ void pack_sig(uint8_t sig[CRYPTO_BYTES],
                                 const uint8_t c_tilde[CTILDEBYTES],
                                 const polyvecl *z, const polyveck *h) {
  unsigned int offset = 0, k = 0;

  for (unsigned int i = 0; i < CTILDEBYTES; ++i) sig[offset++] = c_tilde[i];
  for (unsigned int i = 0; i < PARAM_L; ++i) {
    polyz_pack(sig + offset, &z->vec[i]);
    offset += POLYZ_PACKEDBYTES;
  }

  for (unsigned int i = 0; i < PARAM_OMEGA + PARAM_K; ++i) sig[offset + i] = 0;
  for (unsigned int i = 0; i < PARAM_K; ++i) {
    for (unsigned int j = 0; j < PARAM_N; ++j)
      if (h->vec[i].coeffs[j] != 0) sig[offset + k++] = (uint8_t)j;
    sig[offset + PARAM_OMEGA + i] = (uint8_t)k;
  }
}

static __device__ __noinline__ int unpack_sig(uint8_t c_tilde[CTILDEBYTES],
                                  polyvecl *z, polyveck *h,
                                  const uint8_t sig[CRYPTO_BYTES]) {
  unsigned int offset = 0, k = 0;

  for (unsigned int i = 0; i < CTILDEBYTES; ++i) c_tilde[i] = sig[offset++];
  for (unsigned int i = 0; i < PARAM_L; ++i) {
    polyz_unpack(&z->vec[i], sig + offset);
    offset += POLYZ_PACKEDBYTES;
  }

  for (unsigned int i = 0; i < PARAM_K; ++i) {
    unsigned int prev_k = k;
    for (unsigned int j = 0; j < PARAM_N; ++j) h->vec[i].coeffs[j] = 0;
    unsigned int end = sig[offset + PARAM_OMEGA + i];
    if (end < k || end > PARAM_OMEGA) return 1;
    for (unsigned int j = k; j < end; ++j) {
      if (j > prev_k && sig[offset + j] <= sig[offset + j - 1])
        return 1;
      h->vec[i].coeffs[sig[offset + j]] = 1;
    }
    k = end;
  }
  for (; k < PARAM_OMEGA; ++k) if (sig[offset + k] != 0) return 1;
  return 0;
}

#elif ALGORITHM == ALGO_AIGIS
/* Aigis: z || hint_bitmap || challenge_poly(N/8 + 8 bytes) */
static __device__ void pack_sig(uint8_t sig[CRYPTO_BYTES],
                                 const polyvecl *z, const polyveck *h,
                                 const poly *c) {
  unsigned int i, j, k;
  uint64_t signs, mask;
  unsigned int offset = 0;

  /* z_packed */
  for (i = 0; i < PARAM_L; ++i) {
    polyz_pack(sig + offset, &z->vec[i]);
    offset += POLYZ_PACKEDBYTES;
  }

  /* hint bitmap */
  k = 0;
  for (i = 0; i < PARAM_OMEGA + PARAM_K; ++i) sig[offset + i] = 0;
  for (i = 0; i < PARAM_K; ++i) {
    for (j = 0; j < PARAM_N; ++j)
      if (h->vec[i].coeffs[j] == 1) sig[offset + k++] = (uint8_t)j;
    sig[offset + PARAM_OMEGA + i] = (uint8_t)k;
  }
  offset += PARAM_OMEGA + PARAM_K;

  /* challenge poly: N/8 bytes bitmap + 8 bytes signs */
  signs = 0;
  mask = 1;
  for (i = 0; i < PARAM_N / 8; ++i) {
    sig[offset + i] = 0;
    for (j = 0; j < 8; ++j) {
      if (c->coeffs[8 * i + j] != 0) {
        sig[offset + i] |= (1u << j);
        if (c->coeffs[8 * i + j] == (PARAM_Q - 1)) signs |= mask;
        mask <<= 1;
      }
    }
  }
  offset += PARAM_N / 8;
  for (i = 0; i < 8; ++i) sig[offset + i] = (uint8_t)(signs >> (8 * i));
}

static __device__ __noinline__ int unpack_sig(polyvecl *z, polyveck *h, poly *c,
                                  const uint8_t sig[CRYPTO_BYTES]) {
  unsigned int i, j, k;
  uint64_t signs, mask;
  unsigned int offset = 0;

  /* z_packed */
  for (i = 0; i < PARAM_L; ++i) {
    polyz_unpack(&z->vec[i], sig + offset);
    offset += POLYZ_PACKEDBYTES;
  }

  /* hint bitmap */
  k = 0;
  for (i = 0; i < PARAM_K; ++i) {
    for (j = 0; j < PARAM_N; ++j) h->vec[i].coeffs[j] = 0;
    unsigned int end = sig[offset + PARAM_OMEGA + i];
    if (end < k || end > PARAM_OMEGA) return 1;
    for (j = k; j < end; ++j) {
      if (j > k && sig[offset + j] <= sig[offset + j - 1]) return 1;
      h->vec[i].coeffs[sig[offset + j]] = 1;
    }
    k = end;
  }
  for (j = k; j < PARAM_OMEGA; ++j) if (sig[offset + j]) return 1;
  offset += PARAM_OMEGA + PARAM_K;

  /* challenge poly: N/8 bitmap + 8 signs */
  for (i = 0; i < PARAM_N; ++i) c->coeffs[i] = 0;
  signs = 0;
  for (i = 0; i < 8; ++i)
    signs |= (uint64_t)sig[offset + PARAM_N / 8 + i] << (8 * i);
  mask = 1;
  for (i = 0; i < PARAM_N / 8; ++i) {
    for (j = 0; j < 8; ++j) {
      if ((sig[offset + i] >> j) & 0x01) {
        c->coeffs[8 * i + j] = (signs & mask) ? PARAM_Q - 1 : 1;
        mask <<= 1;
      }
    }
  }
  return 0;
}
#endif

#endif /* PACKING_CUH */
