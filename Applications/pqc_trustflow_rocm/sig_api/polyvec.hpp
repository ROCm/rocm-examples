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

#ifndef POLYVEC_HPP
#define POLYVEC_HPP

#include "params.h"
#include "poly.hpp"

/* Vectors of length L (s1, y, z) */
typedef struct
{
    poly vec[PARAM_L];
} polyvecl;
/* Vectors of length K (s2, t, w, ...) */
typedef struct
{
    poly vec[PARAM_K];
} polyveck;

/* ---------------------------------------------------------------- */
static __device__ void polyvecl_add(polyvecl* w, const polyvecl* u, const polyvecl* v)
{
    for(int i = 0; i < PARAM_L; ++i)
        poly_add(&w->vec[i], &u->vec[i], &v->vec[i]);
}
static __device__ void polyvecl_ntt(polyvecl* v)
{
    for(int i = 0; i < PARAM_L; ++i)
        poly_ntt(&v->vec[i]);
}
static __device__ void polyvecl_invntt_tomont(polyvecl* v)
{
    for(int i = 0; i < PARAM_L; ++i)
        poly_invntt_tomont(&v->vec[i]);
}
static __device__ void
    polyvecl_pointwise_poly_montgomery(polyvecl* r, const poly* a, const polyvecl* v)
{
    for(int i = 0; i < PARAM_L; ++i)
        poly_pointwise_montgomery(&r->vec[i], a, &v->vec[i]);
}
static __device__ void polyvecl_reduce(polyvecl* v)
{
    for(int i = 0; i < PARAM_L; ++i)
        poly_reduce(&v->vec[i]);
}
static __device__ void polyvecl_freeze2q(polyvecl* v)
{
    for(int i = 0; i < PARAM_L; ++i)
        poly_freeze2q(&v->vec[i]);
}
static __device__ void polyvecl_freeze4q(polyvecl* v)
{
    for(int i = 0; i < PARAM_L; ++i)
        poly_freeze4q(&v->vec[i]);
}
static __device__ int polyvecl_chknorm(const polyvecl* v, int32_t bound)
{
    for(int i = 0; i < PARAM_L; ++i)
        if(poly_chknorm(&v->vec[i], bound))
            return 1;
    return 0;
}

/* ---------------------------------------------------------------- */
static __device__ void polyveck_add(polyveck* w, const polyveck* u, const polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_add(&w->vec[i], &u->vec[i], &v->vec[i]);
}
static __device__ void polyveck_sub(polyveck* w, const polyveck* u, const polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_sub(&w->vec[i], &u->vec[i], &v->vec[i]);
}
#if ALGORITHM == ALGO_AIGIS
static __device__ void polyveck_neg(polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_neg(&v->vec[i]);
}
#endif
static __device__ void polyveck_ntt(polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_ntt(&v->vec[i]);
}
static __device__ void polyveck_invntt_tomont(polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_invntt_tomont(&v->vec[i]);
}
static __device__ void
    polyveck_pointwise_poly_montgomery(polyveck* r, const poly* a, const polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_pointwise_montgomery(&r->vec[i], a, &v->vec[i]);
}
static __device__ void polyveck_reduce(polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_reduce(&v->vec[i]);
}
static __device__ void polyveck_caddq(polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_caddq(&v->vec[i]);
}
static __device__ void polyveck_freeze2q(polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_freeze2q(&v->vec[i]);
}
static __device__ void polyveck_freeze4q(polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_freeze4q(&v->vec[i]);
}
static __device__ void polyveck_shiftl(polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_shiftl(&v->vec[i]);
}
static __device__ void polyveck_power2round(polyveck* v1, polyveck* v0, const polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_power2round(&v1->vec[i], &v0->vec[i], &v->vec[i]);
}
static __device__ void polyveck_decompose(polyveck* v1, polyveck* v0, const polyveck* v)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_decompose(&v1->vec[i], &v0->vec[i], &v->vec[i]);
}
static __device__ unsigned int
    polyveck_make_hint(polyveck* h, const polyveck* v0, const polyveck* v1)
{
    unsigned int s = 0;
    for(int i = 0; i < PARAM_K; ++i)
        s += poly_make_hint(&h->vec[i], &v0->vec[i], &v1->vec[i]);
    return s;
}
static __device__ __noinline__ void
    polyveck_use_hint(polyveck* w, const polyveck* v, const polyveck* h)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_use_hint(&w->vec[i], &v->vec[i], &h->vec[i]);
}
static __device__ int polyveck_chknorm(const polyveck* v, int32_t bound)
{
    for(int i = 0; i < PARAM_K; ++i)
        if(poly_chknorm(&v->vec[i], bound))
            return 1;
    return 0;
}

/* ----------------------------------------------------------------
 * Matrix-vector: w = A*v  (both in NTT domain, results accumulated)
 * ---------------------------------------------------------------- */
static __device__ __noinline__ void
    polyveck_accumulate_matvecntt(polyveck* w, const polyvecl row[PARAM_K], const polyvecl* v)
{
    poly t;
    for(int i = 0; i < PARAM_K; ++i)
    {
        poly_pointwise_montgomery(&w->vec[i], &row[i].vec[0], &v->vec[0]);
        for(int j = 1; j < PARAM_L; ++j)
        {
            poly_pointwise_montgomery(&t, &row[i].vec[j], &v->vec[j]);
            poly_add(&w->vec[i], &w->vec[i], &t);
        }
#if ALGORITHM == ALGO_AIGIS
        /* Aigis: accumulated sum in (-L*Q, L*Q); reduce to [0,Q) */
        for(unsigned int c = 0; c < PARAM_N; ++c)
            w->vec[i].coeffs[c] = barrat_reduce(w->vec[i].coeffs[c]);
#endif
    }
}

/* ----------------------------------------------------------------
 * Matrix expansion from rho seed (unified via MATRIX_NONCE macro)
 * ---------------------------------------------------------------- */
static __device__ __noinline__ void polyvec_matrix_expand(polyvecl      mat[PARAM_K],
                                                          const uint8_t rho[SEEDBYTES])
{
    for(int i = 0; i < PARAM_K; ++i)
        for(int j = 0; j < PARAM_L; ++j)
            poly_uniform(&mat[i].vec[j], rho, MATRIX_NONCE(i, j));
}

/* ----------------------------------------------------------------
 * Uniform eta sampling for s1/s2 vectors (unified signature)
 * ---------------------------------------------------------------- */
static __device__ void polyvecl_uniform_eta_s1(polyvecl* v, const uint8_t* seed, uint16_t nonce)
{
    for(int i = 0; i < PARAM_L; ++i)
        poly_uniform_eta_s1(&v->vec[i], seed, nonce++);
}
static __device__ void polyveck_uniform_eta_s2(polyveck* v, const uint8_t* seed, uint16_t nonce)
{
    for(int i = 0; i < PARAM_K; ++i)
        poly_uniform_eta_s2(&v->vec[i], seed, nonce++);
}

/* ----------------------------------------------------------------
 * Uniform gamma1 sampling for y (unified via GAMMA1_NONCE macro)
 * ---------------------------------------------------------------- */
static __device__ void polyvecl_uniform_gamma1(polyvecl* v, const uint8_t* seed, uint16_t nonce)
{
    for(int i = 0; i < PARAM_L; ++i)
        poly_uniform_gamma1(&v->vec[i], seed, GAMMA1_NONCE(nonce, i));
}

/* ----------------------------------------------------------------
 * Pack w1 hint bitmap into flat byte array
 * ---------------------------------------------------------------- */
static __device__ __noinline__ void polyveck_pack_w1(uint8_t r[PARAM_K * POLYW1_PACKEDBYTES],
                                                     const polyveck* w1)
{
    for(int i = 0; i < PARAM_K; ++i)
        polyw1_pack(r + i * POLYW1_PACKEDBYTES, &w1->vec[i]);
}

#endif /* POLYVEC_HPP */
