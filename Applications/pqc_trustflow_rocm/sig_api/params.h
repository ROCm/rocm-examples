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

#ifndef PARAMS_H
#define PARAMS_H

#include <hip/hip_runtime.h>

#include "config.h"
#include <stdint.h>

typedef int32_t coeff_t;
typedef int64_t coeff2_t;

#define PARAM_N 256
#define SEEDBYTES 32

#if ALGORITHM == ALGO_MLDSA

    #define PARAM_Q 8380417
    #define PARAM_QBITS 23
    #define CRHBYTES 64
    #define TRBYTES 64
    #define RNDBYTES 32

    /* Mont constants for Q=8380417:
 *   MONT_VAL = 2^32 mod Q = 4193792
 *   MONT_QINV: Q^{-1} mod 2^32 = 58728449 (fits in uint32) */
    #define MONT_VAL 4193792
    #define MONT_QINV 58728449u

    #if PARAM_MODE == 2 /* ML-DSA-44 */
        #define PARAM_K 4
        #define PARAM_L 4
        #define PARAM_D 13
        #define PARAM_ETA_S1 2
        #define PARAM_ETA_S2 2
        #define PARAM_TAU 39
        #define PARAM_BETA1 78 /* TAU * ETA_S1 */
        #define PARAM_BETA2 78 /* TAU * ETA_S2 */
        #define PARAM_GAMMA1 (1 << 17)
        #define PARAM_GAMMA2 ((PARAM_Q - 1) / 88)
        #define PARAM_OMEGA 80
        #define CTILDEBYTES 32
        #define SETA1BITS 3 /* ceil(log2(2*2+1)) = ceil(log2(5)) = 3 */
        #define SETA2BITS 3
        #define INTT_F 41978 /* N^{-1} * 2^32 mod Q */

    #elif PARAM_MODE == 3 /* ML-DSA-65 */
        #define PARAM_K 6
        #define PARAM_L 5
        #define PARAM_D 13
        #define PARAM_ETA_S1 4
        #define PARAM_ETA_S2 4
        #define PARAM_TAU 49
        #define PARAM_BETA1 196
        #define PARAM_BETA2 196
        #define PARAM_GAMMA1 (1 << 19)
        #define PARAM_GAMMA2 ((PARAM_Q - 1) / 32)
        #define PARAM_OMEGA 55
        #define CTILDEBYTES 48
        #define SETA1BITS 4 /* ceil(log2(2*4+1)) = ceil(log2(9)) = 4 */
        #define SETA2BITS 4
        #define INTT_F 41978

    #elif PARAM_MODE == 5 /* ML-DSA-87 */
        #define PARAM_K 8
        #define PARAM_L 7
        #define PARAM_D 13
        #define PARAM_ETA_S1 2
        #define PARAM_ETA_S2 2
        #define PARAM_TAU 60
        #define PARAM_BETA1 120
        #define PARAM_BETA2 120
        #define PARAM_GAMMA1 (1 << 19)
        #define PARAM_GAMMA2 ((PARAM_Q - 1) / 32)
        #define PARAM_OMEGA 75
        #define CTILDEBYTES 64
        #define SETA1BITS 3
        #define SETA2BITS 3
        #define INTT_F 41978
    #else
        #error "PARAM_MODE must be 2, 3, or 5 for ML-DSA"
    #endif

    #define CRYPTO_ALGNAME "ML-DSA"

#elif ALGORITHM == ALGO_AIGIS

    #define CRHBYTES 48
    #define TRBYTES 48
    #define RNDBYTES 0

    #define PARAM_ALPHA_VAL (2 * ((PARAM_Q - 1) / 12))

    #if PARAM_MODE == 1 /* Aigis-sig1 */
        #define PARAM_Q 2021377
        #define PARAM_QBITS 21
        #define PARAM_K 4
        #define PARAM_L 3
        #define PARAM_D 13
        #define PARAM_ETA_S1 2
        #define PARAM_ETA_S2 3
        #define PARAM_TAU 60
        #define PARAM_BETA1 120 /* TAU * ETA_S1 = 60*2 */
        #define PARAM_BETA2 175 /* from PQMagic params: 175 (~TAU*ETA_S2 slightly adjusted) */
        #define PARAM_GAMMA1 (1 << 17)
        #define PARAM_GAMMA2 ((PARAM_Q - 1) / 12) /* = 168448 */
        #define PARAM_OMEGA 80
        #define SETA1BITS 3 /* ceil(log2(2*2+1))=3 */
        #define SETA2BITS 3 /* ceil(log2(2*3+1))=3 */
        /* Mont: 2^32 mod Q=2021377 = 1562548; Q^{-1} mod 2^32 */
        #define MONT_VAL 1562548
        #define MONT_QINV 1445013505u

    #elif PARAM_MODE == 2 /* Aigis-sig2 */
        #define PARAM_Q 3870721
        #define PARAM_QBITS 22
        #define PARAM_K 5
        #define PARAM_L 4
        #define PARAM_D 14
        #define PARAM_ETA_S1 2
        #define PARAM_ETA_S2 5
        #define PARAM_TAU 60
        #define PARAM_BETA1 120 /* TAU * ETA_S1 = 60*2 */
        #define PARAM_BETA2 275 /* from PQMagic params */
        #define PARAM_GAMMA1 (1 << 17)
        #define PARAM_GAMMA2 ((PARAM_Q - 1) / 12) /* = 322560 */
        #define PARAM_OMEGA 96
        #define SETA1BITS 3 /* ceil(log2(5))=3 */
        #define SETA2BITS 4 /* ceil(log2(11))=4 */
        /* Mont: 2^32 mod Q=3870721 = 2337707; Q^{-1} mod 2^32 */
        #define MONT_VAL 2337707
        #define MONT_QINV 1623519233u

    #elif PARAM_MODE == 3 /* Aigis-sig3 */
        #define PARAM_Q 3870721
        #define PARAM_QBITS 22
        #define PARAM_K 6
        #define PARAM_L 5
        #define PARAM_D 14
        #define PARAM_ETA_S1 1
        #define PARAM_ETA_S2 5
        #define PARAM_TAU 60
        #define PARAM_BETA1 60 /* TAU * ETA_S1 = 60*1 */
        #define PARAM_BETA2 275 /* from PQMagic params */
        #define PARAM_GAMMA1 (1 << 17)
        #define PARAM_GAMMA2 ((PARAM_Q - 1) / 12) /* = 322560 */
        #define PARAM_OMEGA 120
        #define SETA1BITS 2
        #define SETA2BITS 4 /* ceil(log2(11))=4 */
        /* Mont: same as mode 2; Q^{-1} mod 2^32 */
        #define MONT_VAL 2337707
        #define MONT_QINV 1623519233u
    #else
        #error "PARAM_MODE must be 1, 2, or 3 for Aigis-sig"
    #endif

    #define CRYPTO_ALGNAME "Aigis-sig"

#endif /* ALGORITHM */

#if ALGORITHM == ALGO_MLDSA
    #define COEFF_BIAS 0
#elif ALGORITHM == ALGO_AIGIS
    #define COEFF_BIAS PARAM_Q
#endif

#if ALGORITHM == ALGO_MLDSA
    #define MATRIX_NONCE(i, j) ((uint16_t)((i) * 256 + (j)))
#elif ALGORITHM == ALGO_AIGIS
    #define MATRIX_NONCE(i, j) ((uint16_t)((i) + ((j) << 4)))
#endif

#if ALGORITHM == ALGO_MLDSA
    #define GAMMA1_NONCE(base, i) ((uint16_t)(PARAM_L * (base) + (i)))
#elif ALGORITHM == ALGO_AIGIS
    #define GAMMA1_NONCE(base, i) ((uint16_t)((base) + (i)))
#endif

#if ALGORITHM == ALGO_MLDSA
    #define Z_BIAS PARAM_GAMMA1
    #define Z_FIXUP(t) /* nothing */
#elif ALGORITHM == ALGO_AIGIS
    #define Z_BIAS (PARAM_GAMMA1 - 1)
    #define Z_FIXUP(t) (t) += (((int32_t)(t)) >> 31) & PARAM_Q
#endif

/* bits per t1 coeff: POLYT1_PACKED_BITS = QBITS - D */
#define POLYT1_PACKED_BITS (PARAM_QBITS - PARAM_D)
/* bytes per poly t1: N * bits / 8 */
#define POLYT1_PACKEDBYTES (PARAM_N * POLYT1_PACKED_BITS / 8)

/* bytes per poly t0: N * D / 8 */
#define POLYT0_PACKEDBYTES (PARAM_N * PARAM_D / 8)

/* bytes per eta poly (s1): N * SETA1BITS / 8 */
#define POLYETA1_PACKEDBYTES (PARAM_N * SETA1BITS / 8)

/* bytes per eta poly (s2): N * SETA2BITS / 8 */
#define POLYETA2_PACKEDBYTES (PARAM_N * SETA2BITS / 8)

/* bytes per z poly: depends on GAMMA1 (18-bit or 20-bit coeffs) */
#if PARAM_GAMMA1 == (1 << 17)
    #define POLYZ_PACKEDBYTES 576 /* 9 bytes per 4 coeffs (18 bits) */
#elif PARAM_GAMMA1 == (1 << 19)
    #define POLYZ_PACKEDBYTES 640 /* 5 bytes per 2 coeffs (20 bits) */
#endif

#if PARAM_GAMMA2 == (PARAM_Q - 1) / 88
    #define POLYW1_PACKEDBYTES 192
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 32
    #define POLYW1_PACKEDBYTES 128
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 12
    #define POLYW1_PACKEDBYTES 96
#endif

/* Number of distinct high-bit parts: N_W1 = (Q-1) / (2 * GAMMA2) */
#define N_W1 ((PARAM_Q - 1) / (2 * PARAM_GAMMA2))

/* Public/Secret key and Signature sizes */
#define CRYPTO_PUBLICKEYBYTES (SEEDBYTES + PARAM_K * POLYT1_PACKEDBYTES)
#define CRYPTO_SECRETKEYBYTES                                                                  \
    (2 * SEEDBYTES + TRBYTES + PARAM_L * POLYETA1_PACKEDBYTES + PARAM_K * POLYETA2_PACKEDBYTES \
     + PARAM_K * POLYT0_PACKEDBYTES)

#if ALGORITHM == ALGO_MLDSA
    /* ML-DSA sig format: c_tilde || z_packed || hints_bitmap */
    #define CRYPTO_BYTES (CTILDEBYTES + PARAM_L * POLYZ_PACKEDBYTES + PARAM_OMEGA + PARAM_K)
#elif ALGORITHM == ALGO_AIGIS
    /* Aigis sig format: z_packed || hints_bitmap || challenge_poly (N/8+8 bytes) */
    #define CHALLENGE_POLY_PACKEDBYTES (PARAM_N / 8 + 8) /* 40 bytes: bitmap + signs */
    #define CRYPTO_BYTES \
        (PARAM_L * POLYZ_PACKEDBYTES + PARAM_OMEGA + PARAM_K + CHALLENGE_POLY_PACKEDBYTES)
#endif

#endif /* PARAMS_H */
