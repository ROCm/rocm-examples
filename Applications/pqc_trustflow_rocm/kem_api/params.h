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

#include "config.h"
#include <stdint.h>

typedef int16_t coeff_t;

#define PARAM_N 256
#define PARAM_SYMBYTES 32
#define PARAM_SSBYTES 32

#if ALGORITHM == ALGO_KYBER

    #define PARAM_Q 3329
    #define PARAM_QBITS 12
    #define PARAM_QINV 62209 /* Q^{-1} mod 2^16 (used as int16 signed = -3327) */

    #define MONT_R2 1353

    #define PARAM_ETA2 2

    #if PARAM_MODE == 2 /* Kyber512 */
        #define PARAM_K 2
        #define PARAM_ETA1 3
        #define PARAM_BITS_PK 12
        #define PARAM_BITS_C1 10
        #define PARAM_BITS_C2 4
        #define CRYPTO_ALGNAME "Kyber512"

    #elif PARAM_MODE == 3 /* Kyber768 */
        #define PARAM_K 3
        #define PARAM_ETA1 2
        #define PARAM_BITS_PK 12
        #define PARAM_BITS_C1 10
        #define PARAM_BITS_C2 4
        #define CRYPTO_ALGNAME "Kyber768"

    #elif PARAM_MODE == 4 /* Kyber1024 */
        #define PARAM_K 4
        #define PARAM_ETA1 2
        #define PARAM_BITS_PK 12
        #define PARAM_BITS_C1 11
        #define PARAM_BITS_C2 5
        #define CRYPTO_ALGNAME "Kyber1024"

    #else
        #error "PARAM_MODE must be 2, 3, or 4 for Kyber"
    #endif

    #define PARAM_ETA_S PARAM_ETA1
    #define PARAM_ETA_E_KG PARAM_ETA1
    #define PARAM_ETA_E_ENC PARAM_ETA1
    #define PARAM_ETA_E2 PARAM_ETA2

    #define PARAM_POLYBYTES 384

    #define PARAM_PRF_ETA1_BYTES (PARAM_ETA1 * PARAM_N / 4)
    #define PARAM_PRF_ETA2_BYTES (PARAM_ETA2 * PARAM_N / 4)

#elif ALGORITHM == ALGO_AIGIS_ENC

    #define PARAM_Q 7681
    #define PARAM_QBITS 13
    #define PARAM_QINV 57857 /* Q^{-1} mod 2^16 */

    #define MONT_R2 5569 /* R^2 mod Q */

    #if PARAM_MODE == 1 /* Aigis-enc-1 (K=2) */
        #define PARAM_K 2
        #define PARAM_ETA_S 4
        #define PARAM_ETA_E_KG 8
        #define PARAM_ETA_E_ENC 8
        #define PARAM_ETA_E2 8
        #define PARAM_BITS_PK 10
        #define PARAM_BITS_C1 10
        #define PARAM_BITS_C2 3
        #define CRYPTO_ALGNAME "Aigis-enc-1"

    #elif PARAM_MODE == 2 /* Aigis-enc-2 (K=3, low) */
        #define PARAM_K 3
        #define PARAM_ETA_S 1
        #define PARAM_ETA_E_KG 4
        #define PARAM_ETA_E_ENC 4
        #define PARAM_ETA_E2 4
        #define PARAM_BITS_PK 9
        #define PARAM_BITS_C1 9
        #define PARAM_BITS_C2 4
        #define CRYPTO_ALGNAME "Aigis-enc-2"

    #elif PARAM_MODE == 3 /* Aigis-enc-3 (K=3, med) */
        #define PARAM_K 3
        #define PARAM_ETA_S 2
        #define PARAM_ETA_E_KG 4
        #define PARAM_ETA_E_ENC 4
        #define PARAM_ETA_E2 4
        #define PARAM_BITS_PK 10
        #define PARAM_BITS_C1 10
        #define PARAM_BITS_C2 3
        #define CRYPTO_ALGNAME "Aigis-enc-3"

    #elif PARAM_MODE == 4 /* Aigis-enc-4 (K=4, high) */
        #define PARAM_K 4
        #define PARAM_ETA_S 3
        #define PARAM_ETA_E_KG 8
        #define PARAM_ETA_E_ENC 8
        #define PARAM_ETA_E2 8
        #define PARAM_BITS_PK 11
        #define PARAM_BITS_C1 11
        #define PARAM_BITS_C2 5
        #define CRYPTO_ALGNAME "Aigis-enc-4"

    #else
        #error "PARAM_MODE must be 1, 2, 3, or 4 for Aigis-enc"
    #endif

    #define PARAM_POLYBYTES 416

    #define PARAM_PRF_ETA1_BYTES (PARAM_ETA_S * 64)
    #define PARAM_PRF_ETA2_BYTES (PARAM_ETA_E_KG * 64)

#endif /* ALGORITHM */

#define PARAM_POLYVECBYTES (PARAM_K * PARAM_POLYBYTES)
#define PARAM_PK_POLYVEC_BYTES (PARAM_BITS_PK * PARAM_K * PARAM_N / 8)
#define PARAM_CT_VEC_BYTES (PARAM_BITS_C1 * PARAM_K * PARAM_N / 8)
#define PARAM_CT_POLY_BYTES (PARAM_BITS_C2 * PARAM_N / 8)

#define PARAM_PUBLICKEYBYTES (PARAM_PK_POLYVEC_BYTES + PARAM_SYMBYTES)
#define PARAM_INDCPA_SECRETKEYBYTES PARAM_POLYVECBYTES
#define PARAM_SECRETKEYBYTES (PARAM_POLYVECBYTES + PARAM_PUBLICKEYBYTES + 2 * PARAM_SYMBYTES)
#define PARAM_CIPHERTEXTBYTES (PARAM_CT_VEC_BYTES + PARAM_CT_POLY_BYTES)

#define PARAM_GEN_MATRIX_NBLOCKS 4
#define PARAM_XOF_BLOCKBYTES 168 /* SHAKE128_RATE */
#define PARAM_GEN_MATRIX_BUFLEN (PARAM_GEN_MATRIX_NBLOCKS * PARAM_XOF_BLOCKBYTES)

#define MAX_K 4

typedef struct
{
    int16_t coeffs[PARAM_N];
} kem_poly;
typedef struct
{
    kem_poly vec[MAX_K];
} kem_polyvec;

#endif /* PARAMS_H */
