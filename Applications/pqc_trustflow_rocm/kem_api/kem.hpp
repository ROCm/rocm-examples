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

#ifndef KEM_HPP
#define KEM_HPP

#include "cbd.hpp"
#include "fips202.hpp"
#include "ntt.hpp"
#include "params.h"
#include "poly.hpp"
#include "polyvec.hpp"
#include "reduce.hpp"
#include <stdint.h>
#include <string.h>

#ifndef KEM_FORCE_INLINE_PACK
    #define KEM_FORCE_INLINE_PACK 0
#endif

#if KEM_FORCE_INLINE_PACK
    #define KEM_PACK_ATTR __forceinline__
#else
    #define KEM_PACK_ATTR __noinline__
#endif

#ifndef KEM_FORCE_INLINE_TOP
    #define KEM_FORCE_INLINE_TOP 0
#endif

#if KEM_FORCE_INLINE_TOP
    #define KEM_TOP_ATTR __forceinline__
#else
    #define KEM_TOP_ATTR __noinline__
#endif

#ifndef KEM_FAST_DECAP_NO_REENC
    #define KEM_FAST_DECAP_NO_REENC 0
#endif

#if ALGORITHM == ALGO_KYBER

static __device__ unsigned int
    rej_uniform(int16_t* r, unsigned int len, const uint8_t* buf, unsigned int buflen)
{
    unsigned int ctr = 0, pos = 0;
    while(ctr < len && pos + 2 < buflen)
    {
        uint16_t val0 = ((buf[pos + 0]) | ((uint16_t)buf[pos + 1] << 8)) & 0x0FFF;
        uint16_t val1 = ((buf[pos + 1] >> 4) | ((uint16_t)buf[pos + 2] << 4)) & 0x0FFF;
        pos += 3;
        if(val0 < PARAM_Q)
            r[ctr++] = (int16_t)val0;
        if(ctr < len && val1 < PARAM_Q)
            r[ctr++] = (int16_t)val1;
    }
    return ctr;
}

#elif ALGORITHM == ALGO_AIGIS_ENC

static __device__ unsigned int
    rej_uniform(int16_t* r, unsigned int len, const uint8_t* buf, unsigned int buflen)
{
    unsigned int ctr = 0, pos = 0;
    while(ctr < len && pos + 12 < buflen)
    {
        uint16_t v[8];

        v[0] = ((uint16_t)buf[pos + 0] | ((uint16_t)buf[pos + 1] << 8)) & 0x1FFF;
        v[1] = ((uint16_t)buf[pos + 1] >> 5 | ((uint16_t)buf[pos + 2] << 3)
                | ((uint16_t)buf[pos + 3] << 11))
               & 0x1FFF;
        v[2] = ((uint16_t)buf[pos + 3] >> 2 | ((uint16_t)buf[pos + 4] << 6)) & 0x1FFF;
        v[3] = ((uint16_t)buf[pos + 4] >> 7 | ((uint16_t)buf[pos + 5] << 1)
                | ((uint16_t)buf[pos + 6] << 9))
               & 0x1FFF;
        v[4] = ((uint16_t)buf[pos + 6] >> 4 | ((uint16_t)buf[pos + 7] << 4)
                | ((uint16_t)buf[pos + 8] << 12))
               & 0x1FFF;
        v[5] = ((uint16_t)buf[pos + 8] >> 1 | ((uint16_t)buf[pos + 9] << 7)) & 0x1FFF;
        v[6] = ((uint16_t)buf[pos + 9] >> 6 | ((uint16_t)buf[pos + 10] << 2)
                | ((uint16_t)buf[pos + 11] << 10))
               & 0x1FFF;
        v[7] = ((uint16_t)buf[pos + 11] >> 3 | ((uint16_t)buf[pos + 12] << 5)) & 0x1FFF;
        pos += 13;
        for(int i = 0; i < 8 && ctr < len; i++)
        {
            if(v[i] < (uint16_t)PARAM_Q)
                r[ctr++] = (int16_t)v[i];
        }
    }
    return ctr;
}

#endif /* ALGORITHM for rej_uniform */

#ifndef KEM_DIRECT_REJ_UNIFORM
    #define KEM_DIRECT_REJ_UNIFORM 1
#endif

#if KEM_DIRECT_REJ_UNIFORM
typedef struct
{
    uint64_t     s[25];
    unsigned int pos;
} xof_reader;

static __device__ __forceinline__ void
    xof_reader_init(xof_reader* rd, const uint8_t* seed, uint8_t x, uint8_t y)
{
    for(unsigned int i = 0; i < 25; i++)
        rd->s[i] = 0;
    for(unsigned int i = 0; i < PARAM_SYMBYTES; i++)
        rd->s[i >> 3] ^= (uint64_t)seed[i] << (8 * (i & 7));
    rd->s[PARAM_SYMBYTES >> 3] ^= (uint64_t)x << (8 * (PARAM_SYMBYTES & 7));
    rd->s[(PARAM_SYMBYTES + 1) >> 3] ^= (uint64_t)y << (8 * ((PARAM_SYMBYTES + 1) & 7));
    rd->s[(PARAM_SYMBYTES + 2) >> 3] ^= (uint64_t)0x1F << (8 * ((PARAM_SYMBYTES + 2) & 7));
    rd->s[(SHAKE128_RATE - 1) >> 3] ^= 1ULL << 63;
    rd->pos = SHAKE128_RATE;
}

static __device__ __forceinline__ uint8_t xof_reader_u8(xof_reader* rd)
{
    if(rd->pos == SHAKE128_RATE)
    {
        KeccakF1600_StatePermute(rd->s);
        rd->pos = 0;
    }
    uint8_t v = (uint8_t)(rd->s[rd->pos >> 3] >> (8 * (rd->pos & 7)));
    rd->pos++;
    return v;
}

static __device__ __noinline__ void
    rej_uniform_xof(int16_t* r, const uint8_t* seed, uint8_t x, uint8_t y)
{
    xof_reader rd;
    xof_reader_init(&rd, seed, x, y);
    unsigned int ctr = 0;

    #if ALGORITHM == ALGO_KYBER
    while(ctr < PARAM_N)
    {
        uint16_t b0   = xof_reader_u8(&rd);
        uint16_t b1   = xof_reader_u8(&rd);
        uint16_t b2   = xof_reader_u8(&rd);
        uint16_t val0 = (uint16_t)((b0 | (b1 << 8)) & 0x0FFF);
        uint16_t val1 = (uint16_t)(((b1 >> 4) | (b2 << 4)) & 0x0FFF);
        if(val0 < PARAM_Q)
            r[ctr++] = (int16_t)val0;
        if(ctr < PARAM_N && val1 < PARAM_Q)
            r[ctr++] = (int16_t)val1;
    }
    #elif ALGORITHM == ALGO_AIGIS_ENC
    while(ctr < PARAM_N)
    {
        uint8_t  b0  = xof_reader_u8(&rd);
        uint8_t  b1  = xof_reader_u8(&rd);
        uint8_t  b2  = xof_reader_u8(&rd);
        uint8_t  b3  = xof_reader_u8(&rd);
        uint8_t  b4  = xof_reader_u8(&rd);
        uint8_t  b5  = xof_reader_u8(&rd);
        uint8_t  b6  = xof_reader_u8(&rd);
        uint8_t  b7  = xof_reader_u8(&rd);
        uint8_t  b8  = xof_reader_u8(&rd);
        uint8_t  b9  = xof_reader_u8(&rd);
        uint8_t  b10 = xof_reader_u8(&rd);
        uint8_t  b11 = xof_reader_u8(&rd);
        uint8_t  b12 = xof_reader_u8(&rd);
        uint16_t v0  = ((uint16_t)b0 | ((uint16_t)b1 << 8)) & 0x1FFF;
        uint16_t v1  = ((uint16_t)b1 >> 5 | ((uint16_t)b2 << 3) | ((uint16_t)b3 << 11)) & 0x1FFF;
        uint16_t v2  = ((uint16_t)b3 >> 2 | ((uint16_t)b4 << 6)) & 0x1FFF;
        uint16_t v3  = ((uint16_t)b4 >> 7 | ((uint16_t)b5 << 1) | ((uint16_t)b6 << 9)) & 0x1FFF;
        uint16_t v4  = ((uint16_t)b6 >> 4 | ((uint16_t)b7 << 4) | ((uint16_t)b8 << 12)) & 0x1FFF;
        uint16_t v5  = ((uint16_t)b8 >> 1 | ((uint16_t)b9 << 7)) & 0x1FFF;
        uint16_t v6  = ((uint16_t)b9 >> 6 | ((uint16_t)b10 << 2) | ((uint16_t)b11 << 10)) & 0x1FFF;
        uint16_t v7  = ((uint16_t)b11 >> 3 | ((uint16_t)b12 << 5)) & 0x1FFF;
        if(v0 < PARAM_Q)
            r[ctr++] = (int16_t)v0;
        if(ctr < PARAM_N && v1 < PARAM_Q)
            r[ctr++] = (int16_t)v1;
        if(ctr < PARAM_N && v2 < PARAM_Q)
            r[ctr++] = (int16_t)v2;
        if(ctr < PARAM_N && v3 < PARAM_Q)
            r[ctr++] = (int16_t)v3;
        if(ctr < PARAM_N && v4 < PARAM_Q)
            r[ctr++] = (int16_t)v4;
        if(ctr < PARAM_N && v5 < PARAM_Q)
            r[ctr++] = (int16_t)v5;
        if(ctr < PARAM_N && v6 < PARAM_Q)
            r[ctr++] = (int16_t)v6;
        if(ctr < PARAM_N && v7 < PARAM_Q)
            r[ctr++] = (int16_t)v7;
    }
    #endif
}
#endif

static __device__ __noinline__ void gen_matrix(kem_polyvec* a, const uint8_t* seed, int transposed)
{
#if KEM_DIRECT_REJ_UNIFORM
    for(int i = 0; i < PARAM_K; i++)
    {
        for(int j = 0; j < PARAM_K; j++)
        {
            uint8_t x, y;
    #if ALGORITHM == ALGO_KYBER
            if(transposed)
            {
                x = (uint8_t)j;
                y = (uint8_t)i;
            }
            else
            {
                x = (uint8_t)i;
                y = (uint8_t)j;
            }
    #elif ALGORITHM == ALGO_AIGIS_ENC
            if(transposed)
            {
                x = (uint8_t)j;
                y = (uint8_t)i;
            }
            else
            {
                x = (uint8_t)i;
                y = (uint8_t)j;
            }
    #endif
            rej_uniform_xof(a[i].vec[j].coeffs, seed, x, y);
        }
    }
#else
    keccak_state state;
    uint8_t      buf[PARAM_GEN_MATRIX_BUFLEN + 2];
    unsigned int ctr;

    for(int i = 0; i < PARAM_K; i++)
    {
        for(int j = 0; j < PARAM_K; j++)
        {

            uint8_t extseed[PARAM_SYMBYTES + 2];
            for(int k = 0; k < PARAM_SYMBYTES; k++)
                extseed[k] = seed[k];

    #if ALGORITHM == ALGO_KYBER

            if(transposed)
            {
                extseed[PARAM_SYMBYTES]     = (uint8_t)j;
                extseed[PARAM_SYMBYTES + 1] = (uint8_t)i;
            }
            else
            {
                extseed[PARAM_SYMBYTES]     = (uint8_t)i;
                extseed[PARAM_SYMBYTES + 1] = (uint8_t)j;
            }
    #elif ALGORITHM == ALGO_AIGIS_ENC

            if(transposed)
            {
                extseed[PARAM_SYMBYTES]     = (uint8_t)j;
                extseed[PARAM_SYMBYTES + 1] = (uint8_t)i;
            }
            else
            {
                extseed[PARAM_SYMBYTES]     = (uint8_t)i;
                extseed[PARAM_SYMBYTES + 1] = (uint8_t)j;
            }
    #endif

            shake128_absorb_once(&state, extseed, PARAM_SYMBYTES + 2);

            ctr = 0;
            while(ctr < PARAM_N)
            {
                shake128_squeezeblocks(buf, PARAM_GEN_MATRIX_NBLOCKS, &state);
                ctr += rej_uniform(a[i].vec[j].coeffs + ctr,
                                   PARAM_N - ctr,
                                   buf,
                                   PARAM_GEN_MATRIX_NBLOCKS * PARAM_XOF_BLOCKBYTES);
            }
        }
    }
#endif
}

static __device__ __noinline__ void
    gen_matrix_row(kem_polyvec* rowvec, const uint8_t* seed, int row, int transposed)
{
#if KEM_DIRECT_REJ_UNIFORM
    for(int j = 0; j < PARAM_K; j++)
    {
        uint8_t x, y;
    #if ALGORITHM == ALGO_KYBER
        if(transposed)
        {
            x = (uint8_t)j;
            y = (uint8_t)row;
        }
        else
        {
            x = (uint8_t)row;
            y = (uint8_t)j;
        }
    #elif ALGORITHM == ALGO_AIGIS_ENC
        if(transposed)
        {
            x = (uint8_t)j;
            y = (uint8_t)row;
        }
        else
        {
            x = (uint8_t)row;
            y = (uint8_t)j;
        }
    #endif
        rej_uniform_xof(rowvec->vec[j].coeffs, seed, x, y);
    }
#else
    kem_polyvec mat[PARAM_K];
    gen_matrix(mat, seed, transposed);
    for(int j = 0; j < PARAM_K; j++)
        for(int c = 0; c < PARAM_N; c++)
            rowvec->vec[j].coeffs[c] = mat[row].vec[j].coeffs[c];
#endif
}

/* pk = pk_vec_bytes || rho */
static __device__ KEM_PACK_ATTR void
    pack_pk(uint8_t* pk, const kem_polyvec* pkpv, const uint8_t* rho)
{
    polyvec_pk_compress(pk, pkpv);
    for(int i = 0; i < PARAM_SYMBYTES; i++)
        pk[PARAM_PK_POLYVEC_BYTES + i] = rho[i];
}

static __device__ KEM_PACK_ATTR void unpack_pk(kem_polyvec* pkpv, uint8_t* rho, const uint8_t* pk)
{
    polyvec_pk_decompress(pkpv, pk);
    for(int i = 0; i < PARAM_SYMBYTES; i++)
        rho[i] = pk[PARAM_PK_POLYVEC_BYTES + i];
}

static __device__ KEM_PACK_ATTR void pack_sk(uint8_t* sk, const kem_polyvec* skpv)
{
    polyvec_tobytes(sk, skpv);
}

static __device__ KEM_PACK_ATTR void unpack_sk(kem_polyvec* skpv, const uint8_t* sk)
{
    polyvec_frombytes(skpv, sk);
}

/* ct = ct_vec_bytes || ct_poly_bytes */
static __device__ KEM_PACK_ATTR void
    pack_ciphertext(uint8_t* c, const kem_polyvec* b, const kem_poly* v)
{
    polyvec_ct_compress(c, b);
    poly_compress_c2(c + PARAM_CT_VEC_BYTES, v);
}

static __device__ KEM_PACK_ATTR void
    unpack_ciphertext(kem_polyvec* b, kem_poly* v, const uint8_t* c)
{
    polyvec_ct_decompress(b, c);
    poly_decompress_c2(v, c + PARAM_CT_VEC_BYTES);
}

static __device__ KEM_TOP_ATTR void indcpa_keypair(uint8_t* pk, uint8_t* sk, const uint8_t* coins)
{
    kem_polyvec    arow, skpv, e, pkpv;
    uint8_t        buf[2 * PARAM_SYMBYTES];
    const uint8_t* publicseed = buf;
    const uint8_t* noiseseed  = buf + PARAM_SYMBYTES;
    uint8_t        nonce      = 0;

    sha3_512(buf, coins, PARAM_SYMBYTES);

    for(int i = 0; i < PARAM_K; i++)
        poly_getnoise_s(skpv.vec[i].coeffs, noiseseed, nonce++);
    for(int i = 0; i < PARAM_K; i++)
        poly_getnoise_e_kg(e.vec[i].coeffs, noiseseed, nonce++);

    /* NTT(s) */
    polyvec_ntt(&skpv);
    polyvec_caddq(&skpv);

    for(int i = 0; i < PARAM_K; i++)
    {
        gen_matrix_row(&arow, publicseed, i, 0 /* not transposed */);
        polyvec_basemul_acc(&pkpv.vec[i], &arow, &skpv);
    }
    polyvec_invntt(&pkpv);
    polyvec_add(&pkpv, &pkpv, &e);
    polyvec_caddq(&pkpv);

    pack_sk(sk, &skpv);
    pack_pk(pk, &pkpv, publicseed);
}

static __device__ KEM_TOP_ATTR void
    indcpa_enc(uint8_t* c, const uint8_t* m, const uint8_t* pk, const uint8_t* coins)
{
    kem_polyvec at[PARAM_K], sp, ep, pkpv, b;
    kem_poly    epp, v, k;
    uint8_t     rho[PARAM_SYMBYTES];
    uint8_t     nonce = 0;

    unpack_pk(&pkpv, rho, pk);
    poly_frommsg(&k, m);
    gen_matrix(at, rho, 1 /* transposed: A^T */);

    for(int i = 0; i < PARAM_K; i++)
        poly_getnoise_s(sp.vec[i].coeffs, coins, nonce++);
    for(int i = 0; i < PARAM_K; i++)
        poly_getnoise_e_enc(ep.vec[i].coeffs, coins, nonce++);
    poly_getnoise_e2(epp.coeffs, coins, nonce++);

    polyvec_ntt(&sp);
    polyvec_ntt(&pkpv);

    for(int i = 0; i < PARAM_K; i++)
    {
        polyvec_basemul_acc(&b.vec[i], &at[i], &sp);
    }
    polyvec_invntt(&b);
    polyvec_add(&b, &b, &ep);
    polyvec_caddq(&b);

    polyvec_basemul_acc(&v, &pkpv, &sp);
    poly_invntt(&v);
    poly_add(&v, &v, &epp);

#if ALGORITHM == ALGO_KYBER
    /* Kyber: v += msg */
    poly_add(&v, &v, &k);
    poly_caddq(&v);
#elif ALGORITHM == ALGO_AIGIS_ENC
    /* Aigis: v -= msg */
    poly_sub(&v, &v, &k);
    poly_caddq2(&v);
#endif

    pack_ciphertext(c, &b, &v);
}

static __device__ KEM_TOP_ATTR void indcpa_dec(uint8_t* m, const uint8_t* c, const uint8_t* sk)
{
    kem_polyvec b, skpv;
    kem_poly    v, mp;

    unpack_ciphertext(&b, &v, c);
    unpack_sk(&skpv, sk);

    polyvec_ntt(&b);
    polyvec_basemul_acc(&mp, &skpv, &b);
    poly_invntt(&mp);

    /* mp = s^T * u - v */
    poly_sub(&mp, &mp, &v);
    poly_caddq2(&mp);

    poly_tomsg(m, &mp);
}

static __device__ KEM_TOP_ATTR void kem_keypair(uint8_t* pk, uint8_t* sk, const uint8_t* coins)
{
    uint8_t coins_indcpa[PARAM_SYMBYTES];

    for(int i = 0; i < PARAM_SYMBYTES; i++)
        coins_indcpa[i] = coins[i];

    indcpa_keypair(pk, sk, coins_indcpa);

    /* sk[INDCPA_SK] = indcpa_sk, sk[INDCPA_SK+PK] = pk */
    uint8_t* sk_pk = sk + PARAM_INDCPA_SECRETKEYBYTES;
    for(int i = 0; i < (int)PARAM_PUBLICKEYBYTES; i++)
        sk_pk[i] = pk[i];

    uint8_t* hpk = sk + PARAM_INDCPA_SECRETKEYBYTES + PARAM_PUBLICKEYBYTES;
    sha3_256(hpk, pk, PARAM_PUBLICKEYBYTES);

    /* z = random (coins[32:64]) */
    uint8_t* z = hpk + PARAM_SYMBYTES;
    for(int i = 0; i < PARAM_SYMBYTES; i++)
        z[i] = coins[PARAM_SYMBYTES + i];
}

static __device__ KEM_TOP_ATTR void
    kem_encaps(uint8_t* ct, uint8_t* ss, const uint8_t* pk, const uint8_t* coins)
{
    uint8_t buf[2 * PARAM_SYMBYTES];
    uint8_t kr[2 * PARAM_SYMBYTES];

    for(int i = 0; i < PARAM_SYMBYTES; i++)
        buf[i] = coins[i];
    sha3_256(buf + PARAM_SYMBYTES, pk, PARAM_PUBLICKEYBYTES);
    sha3_512(kr, buf, 2 * PARAM_SYMBYTES);

    indcpa_enc(ct, buf, pk, kr + PARAM_SYMBYTES);

    /* ss = SHAKE256(K' || H(ct)) */
    sha3_256(kr + PARAM_SYMBYTES, ct, PARAM_CIPHERTEXTBYTES);
    shake256(ss, PARAM_SSBYTES, kr, 2 * PARAM_SYMBYTES);
}

static __device__ KEM_TOP_ATTR void kem_decaps(uint8_t* ss, const uint8_t* ct, const uint8_t* sk)
{
    const uint8_t* pk  = sk + PARAM_INDCPA_SECRETKEYBYTES;
    const uint8_t* hpk = pk + PARAM_PUBLICKEYBYTES;
    const uint8_t* z   = hpk + PARAM_SYMBYTES;

    uint8_t buf[2 * PARAM_SYMBYTES];
    uint8_t kr[2 * PARAM_SYMBYTES];
#if !KEM_FAST_DECAP_NO_REENC
    uint8_t ct_reenc[PARAM_CIPHERTEXTBYTES];
#endif

    indcpa_dec(buf, ct, sk);

    for(int i = 0; i < PARAM_SYMBYTES; i++)
        buf[PARAM_SYMBYTES + i] = hpk[i];
    sha3_512(kr, buf, 2 * PARAM_SYMBYTES);

#if !KEM_FAST_DECAP_NO_REENC
    indcpa_enc(ct_reenc, buf, pk, kr + PARAM_SYMBYTES);

    int diff = 0;
    for(int i = 0; i < (int)PARAM_CIPHERTEXTBYTES; i++)
        diff |= (ct[i] ^ ct_reenc[i]);

    uint8_t fail = (uint8_t)(0u - (unsigned)(diff != 0));

    sha3_256(kr + PARAM_SYMBYTES, ct, PARAM_CIPHERTEXTBYTES);

    for(int i = 0; i < PARAM_SYMBYTES; i++)
        kr[i] = (uint8_t)((kr[i] & ~fail) | (z[i] & fail));
#else
    (void)pk;
    (void)z;
#endif

    shake256(ss, PARAM_SSBYTES, kr, 2 * PARAM_SYMBYTES);
}

#endif /* KEM_HPP */
