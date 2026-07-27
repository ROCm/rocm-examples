// MIT License
//
// Copyright (c) 2026 firedoil
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SYMMETRIC_HPP
#define SYMMETRIC_HPP

#include "fips202.hpp"
#include "params.h"
#include <stdint.h>

typedef keccak_state stream128_state;
typedef keccak_state stream256_state;

#define STREAM128_BLOCKBYTES SHAKE128_RATE
#define STREAM256_BLOCKBYTES SHAKE256_RATE

#if ALGORITHM == ALGO_MLDSA
/* ---- ML-DSA: SEEDBYTES seed + 2-byte nonce (stream128)
 *              CRHBYTES seed + 2-byte nonce (stream256) ---- */

static __device__ void dilithium_shake128_stream_init(keccak_state* state,
                                                      const uint8_t seed[SEEDBYTES],
                                                      uint16_t      nonce)
{
    uint8_t t[2];
    t[0] = nonce;
    t[1] = nonce >> 8;
    shake128_init(state);
    shake128_absorb(state, seed, SEEDBYTES);
    shake128_absorb(state, t, 2);
    shake128_finalize(state);
}

static __device__ void dilithium_shake256_stream_init(keccak_state* state,
                                                      const uint8_t seed[CRHBYTES],
                                                      uint16_t      nonce)
{
    uint8_t t[2];
    t[0] = nonce;
    t[1] = nonce >> 8;
    shake256_init(state);
    shake256_absorb(state, seed, CRHBYTES);
    shake256_absorb(state, t, 2);
    shake256_finalize(state);
}

    #define stream128_init(STATE, SEED, NONCE) dilithium_shake128_stream_init(STATE, SEED, NONCE)
    #define stream128_squeezeblocks(OUT, OUTBLOCKS, STATE) \
        shake128_squeezeblocks(OUT, OUTBLOCKS, STATE)
    #define stream256_init(STATE, SEED, NONCE) dilithium_shake256_stream_init(STATE, SEED, NONCE)
    #define stream256_squeezeblocks(OUT, OUTBLOCKS, STATE) \
        shake256_squeezeblocks(OUT, OUTBLOCKS, STATE)

/* These Aigis-named shims keep inactive template bodies valid for HIP clang. */
static __device__ void
    aigis_shake128_stream_init(keccak_state* state, const uint8_t seed[SEEDBYTES], uint8_t nonce)
{
    dilithium_shake128_stream_init(state, seed, (uint16_t)nonce);
}
static __device__ void
    aigis_shake256_eta_init(keccak_state* state, const uint8_t seed[SEEDBYTES], uint8_t nonce)
{
    (void)nonce;
    shake256_init(state);
    shake256_absorb(state, seed, SEEDBYTES);
    shake256_finalize(state);
}
static __device__ void aigis_shake256_gamma1_init(keccak_state* state,
                                                  const uint8_t seed[SEEDBYTES + CRHBYTES],
                                                  uint16_t      nonce)
{
    dilithium_shake256_stream_init(state, seed, nonce);
}
#elif ALGORITHM == ALGO_AIGIS
/* ---- Aigis: matrix A expand  = SEEDBYTES + 1-byte nonce via shake128
 *            eta sampling      = SEEDBYTES + 1-byte nonce via shake256
 *            gamma1 sampling   = (SEEDBYTES+CRHBYTES) + 2-byte nonce via shake256 ---- */

/* Matrix A: shake128(seed || 1-byte nonce) */
static __device__ void
    aigis_shake128_stream_init(keccak_state* state, const uint8_t seed[SEEDBYTES], uint8_t nonce)
{
    shake128_init(state);
    shake128_absorb(state, seed, SEEDBYTES);
    shake128_absorb(state, &nonce, 1);
    shake128_finalize(state);
}

/* Eta sampling: shake256(seed || 1-byte nonce) */
static __device__ void
    aigis_shake256_eta_init(keccak_state* state, const uint8_t seed[SEEDBYTES], uint8_t nonce)
{
    shake256_init(state);
    shake256_absorb(state, seed, SEEDBYTES);
    shake256_absorb(state, &nonce, 1);
    shake256_finalize(state);
}

/* Gamma1 sampling: shake256(seed(SEEDBYTES+CRHBYTES) || 2-byte nonce) */
static __device__ void aigis_shake256_gamma1_init(keccak_state* state,
                                                  const uint8_t seed[SEEDBYTES + CRHBYTES],
                                                  uint16_t      nonce)
{
    uint8_t t[2];
    t[0] = nonce & 0xFF;
    t[1] = nonce >> 8;
    shake256_init(state);
    shake256_absorb(state, seed, SEEDBYTES + CRHBYTES);
    shake256_absorb(state, t, 2);
    shake256_finalize(state);
}

    #define stream128_squeezeblocks(OUT, OUTBLOCKS, STATE) \
        shake128_squeezeblocks(OUT, OUTBLOCKS, STATE)
    #define stream256_squeezeblocks(OUT, OUTBLOCKS, STATE) \
        shake256_squeezeblocks(OUT, OUTBLOCKS, STATE)

    /* These aliases keep inactive ML-DSA template bodies valid for HIP clang. */
    #define stream128_init(STATE, SEED, NONCE) \
        aigis_shake128_stream_init(STATE, SEED, (uint8_t)(NONCE))
    #define stream256_init(STATE, SEED, NONCE) \
        aigis_shake256_gamma1_init(STATE, (const uint8_t*)(SEED), (uint16_t)(NONCE))

#endif /* ALGORITHM */

#endif
