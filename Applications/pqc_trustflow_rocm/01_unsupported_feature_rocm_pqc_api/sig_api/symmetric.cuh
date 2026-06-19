#ifndef SYMMETRIC_CUH
#define SYMMETRIC_CUH

#include <stdint.h>
#include "params.h"
#include "fips202.cuh"

typedef keccak_state stream128_state;
typedef keccak_state stream256_state;

#define STREAM128_BLOCKBYTES SHAKE128_RATE
#define STREAM256_BLOCKBYTES SHAKE256_RATE

#if ALGORITHM == ALGO_MLDSA
/* ---- ML-DSA: SEEDBYTES seed + 2-byte nonce (stream128)
 *              CRHBYTES seed + 2-byte nonce (stream256) ---- */

static __device__ void dilithium_shake128_stream_init(keccak_state *state, const uint8_t seed[SEEDBYTES], uint16_t nonce) {
  uint8_t t[2];
  t[0] = nonce;
  t[1] = nonce >> 8;
  shake128_init(state);
  shake128_absorb(state, seed, SEEDBYTES);
  shake128_absorb(state, t, 2);
  shake128_finalize(state);
}

static __device__ void dilithium_shake256_stream_init(keccak_state *state, const uint8_t seed[CRHBYTES], uint16_t nonce) {
  uint8_t t[2];
  t[0] = nonce;
  t[1] = nonce >> 8;
  shake256_init(state);
  shake256_absorb(state, seed, CRHBYTES);
  shake256_absorb(state, t, 2);
  shake256_finalize(state);
}

#define stream128_init(STATE, SEED, NONCE) dilithium_shake128_stream_init(STATE, SEED, NONCE)
#define stream128_squeezeblocks(OUT, OUTBLOCKS, STATE) shake128_squeezeblocks(OUT, OUTBLOCKS, STATE)
#define stream256_init(STATE, SEED, NONCE) dilithium_shake256_stream_init(STATE, SEED, NONCE)
#define stream256_squeezeblocks(OUT, OUTBLOCKS, STATE) shake256_squeezeblocks(OUT, OUTBLOCKS, STATE)


/* HIP clang parses non-instantiated template bodies more strictly than NVCC.
 * These Aigis-named shims are only visible while compiling ML-DSA. */
static __device__ void aigis_shake128_stream_init(keccak_state *state, const uint8_t seed[SEEDBYTES], uint8_t nonce) {
  dilithium_shake128_stream_init(state, seed, (uint16_t)nonce);
}
static __device__ void aigis_shake256_eta_init(keccak_state *state, const uint8_t seed[SEEDBYTES], uint8_t nonce) {
  (void)nonce;
  shake256_init(state);
  shake256_absorb(state, seed, SEEDBYTES);
  shake256_finalize(state);
}
static __device__ void aigis_shake256_gamma1_init(keccak_state *state, const uint8_t seed[SEEDBYTES + CRHBYTES], uint16_t nonce) {
  dilithium_shake256_stream_init(state, seed, nonce);
}
#elif ALGORITHM == ALGO_AIGIS
/* ---- Aigis: matrix A expand  = SEEDBYTES + 1-byte nonce via shake128
 *            eta sampling      = SEEDBYTES + 1-byte nonce via shake256
 *            gamma1 sampling   = (SEEDBYTES+CRHBYTES) + 2-byte nonce via shake256 ---- */

/* Matrix A: shake128(seed || 1-byte nonce) */
static __device__ void aigis_shake128_stream_init(keccak_state *state, const uint8_t seed[SEEDBYTES], uint8_t nonce) {
  shake128_init(state);
  shake128_absorb(state, seed, SEEDBYTES);
  shake128_absorb(state, &nonce, 1);
  shake128_finalize(state);
}

/* Eta sampling: shake256(seed || 1-byte nonce) */
static __device__ void aigis_shake256_eta_init(keccak_state *state, const uint8_t seed[SEEDBYTES], uint8_t nonce) {
  shake256_init(state);
  shake256_absorb(state, seed, SEEDBYTES);
  shake256_absorb(state, &nonce, 1);
  shake256_finalize(state);
}

/* Gamma1 sampling: shake256(seed(SEEDBYTES+CRHBYTES) || 2-byte nonce) */
static __device__ void aigis_shake256_gamma1_init(keccak_state *state,
                                                   const uint8_t seed[SEEDBYTES + CRHBYTES],
                                                   uint16_t nonce) {
  uint8_t t[2];
  t[0] = nonce & 0xFF;
  t[1] = nonce >> 8;
  shake256_init(state);
  shake256_absorb(state, seed, SEEDBYTES + CRHBYTES);
  shake256_absorb(state, t, 2);
  shake256_finalize(state);
}

/* Aigis stream macros — these are NOT exact aliases of the ML-DSA ones;
 * callers that differ (matrix, eta, gamma1) call the specific inits above. */
#define stream128_squeezeblocks(OUT, OUTBLOCKS, STATE) shake128_squeezeblocks(OUT, OUTBLOCKS, STATE)
#define stream256_squeezeblocks(OUT, OUTBLOCKS, STATE) shake256_squeezeblocks(OUT, OUTBLOCKS, STATE)

/* HIP clang parses non-instantiated ML-DSA template bodies while compiling Aigis.
 * These stream_init aliases are only for parsing; active Aigis paths call the
 * Aigis-specific init functions above. */
#define stream128_init(STATE, SEED, NONCE) aigis_shake128_stream_init(STATE, SEED, (uint8_t)(NONCE))
#define stream256_init(STATE, SEED, NONCE) aigis_shake256_gamma1_init(STATE, (const uint8_t *)(SEED), (uint16_t)(NONCE))

#endif /* ALGORITHM */

#endif
