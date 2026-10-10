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

#ifndef FIPS202_HPP
#define FIPS202_HPP

#include <stdint.h>

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_512_RATE 72
#define NROUNDS 24
#define ROL(a, offset) (((a) << (offset)) ^ ((a) >> (64 - (offset))))

typedef struct
{
    uint64_t     s[25];
    unsigned int pos;
} keccak_state;

__constant__ uint64_t gpu_KeccakF_RoundConstants[NROUNDS]
    = {0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
       0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
       0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
       0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
       0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
       0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

static __device__ __forceinline__ uint64_t gpu_load64(const uint8_t* x)
{
    uint64_t r = 0;
    for(int i = 0; i < 8; ++i)
        r |= (uint64_t)x[i] << (8 * i);
    return r;
}

static __device__ __forceinline__ void gpu_store64(uint8_t* x, uint64_t u)
{
    for(unsigned int i = 0; i < 8; ++i)
    {
        x[i] = (uint8_t)u;
        u >>= 8;
    }
}

static __device__ __noinline__ void KeccakF1600_StatePermute(uint64_t* state)
{
    int      round;
    uint64_t Aba, Abe, Abi, Abo, Abu;
    uint64_t Aga, Age, Agi, Ago, Agu;
    uint64_t Aka, Ake, Aki, Ako, Aku;
    uint64_t Ama, Ame, Ami, Amo, Amu;
    uint64_t Asa, Ase, Asi, Aso, Asu;
    uint64_t BCa, BCe, BCi, BCo, BCu;
    uint64_t Da, De, Di, Do, Du;
    uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
    uint64_t Ega, Ege, Egi, Ego, Egu;
    uint64_t Eka, Eke, Eki, Eko, Eku;
    uint64_t Ema, Eme, Emi, Emo, Emu;
    uint64_t Esa, Ese, Esi, Eso, Esu;

    Aba = state[0];
    Abe = state[1];
    Abi = state[2];
    Abo = state[3];
    Abu = state[4];
    Aga = state[5];
    Age = state[6];
    Agi = state[7];
    Ago = state[8];
    Agu = state[9];
    Aka = state[10];
    Ake = state[11];
    Aki = state[12];
    Ako = state[13];
    Aku = state[14];
    Ama = state[15];
    Ame = state[16];
    Ami = state[17];
    Amo = state[18];
    Amu = state[19];
    Asa = state[20];
    Ase = state[21];
    Asi = state[22];
    Aso = state[23];
    Asu = state[24];

    for(round = 0; round < NROUNDS; round += 2)
    {
        BCa = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        BCe = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        BCi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        BCo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        BCu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;
        Da  = BCu ^ ROL(BCe, 1);
        De  = BCa ^ ROL(BCi, 1);
        Di  = BCe ^ ROL(BCo, 1);
        Do  = BCi ^ ROL(BCu, 1);
        Du  = BCo ^ ROL(BCa, 1);

        Aba ^= Da;
        BCa = Aba;
        Age ^= De;
        BCe = ROL(Age, 44);
        Aki ^= Di;
        BCi = ROL(Aki, 43);
        Amo ^= Do;
        BCo = ROL(Amo, 21);
        Asu ^= Du;
        BCu = ROL(Asu, 14);
        Eba = BCa ^ ((~BCe) & BCi);
        Eba ^= gpu_KeccakF_RoundConstants[round];
        Ebe = BCe ^ ((~BCi) & BCo);
        Ebi = BCi ^ ((~BCo) & BCu);
        Ebo = BCo ^ ((~BCu) & BCa);
        Ebu = BCu ^ ((~BCa) & BCe);

        Abo ^= Do;
        BCa = ROL(Abo, 28);
        Agu ^= Du;
        BCe = ROL(Agu, 20);
        Aka ^= Da;
        BCi = ROL(Aka, 3);
        Ame ^= De;
        BCo = ROL(Ame, 45);
        Asi ^= Di;
        BCu = ROL(Asi, 61);
        Ega = BCa ^ ((~BCe) & BCi);
        Ege = BCe ^ ((~BCi) & BCo);
        Egi = BCi ^ ((~BCo) & BCu);
        Ego = BCo ^ ((~BCu) & BCa);
        Egu = BCu ^ ((~BCa) & BCe);

        Abe ^= De;
        BCa = ROL(Abe, 1);
        Agi ^= Di;
        BCe = ROL(Agi, 6);
        Ako ^= Do;
        BCi = ROL(Ako, 25);
        Amu ^= Du;
        BCo = ROL(Amu, 8);
        Asa ^= Da;
        BCu = ROL(Asa, 18);
        Eka = BCa ^ ((~BCe) & BCi);
        Eke = BCe ^ ((~BCi) & BCo);
        Eki = BCi ^ ((~BCo) & BCu);
        Eko = BCo ^ ((~BCu) & BCa);
        Eku = BCu ^ ((~BCa) & BCe);

        Abu ^= Du;
        BCa = ROL(Abu, 27);
        Aga ^= Da;
        BCe = ROL(Aga, 36);
        Ake ^= De;
        BCi = ROL(Ake, 10);
        Ami ^= Di;
        BCo = ROL(Ami, 15);
        Aso ^= Do;
        BCu = ROL(Aso, 56);
        Ema = BCa ^ ((~BCe) & BCi);
        Eme = BCe ^ ((~BCi) & BCo);
        Emi = BCi ^ ((~BCo) & BCu);
        Emo = BCo ^ ((~BCu) & BCa);
        Emu = BCu ^ ((~BCa) & BCe);

        Abi ^= Di;
        BCa = ROL(Abi, 62);
        Ago ^= Do;
        BCe = ROL(Ago, 55);
        Aku ^= Du;
        BCi = ROL(Aku, 39);
        Ama ^= Da;
        BCo = ROL(Ama, 41);
        Ase ^= De;
        BCu = ROL(Ase, 2);
        Esa = BCa ^ ((~BCe) & BCi);
        Ese = BCe ^ ((~BCi) & BCo);
        Esi = BCi ^ ((~BCo) & BCu);
        Eso = BCo ^ ((~BCu) & BCa);
        Esu = BCu ^ ((~BCa) & BCe);

        /* Round 2 */
        BCa = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        BCe = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        BCi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        BCo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        BCu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;
        Da  = BCu ^ ROL(BCe, 1);
        De  = BCa ^ ROL(BCi, 1);
        Di  = BCe ^ ROL(BCo, 1);
        Do  = BCi ^ ROL(BCu, 1);
        Du  = BCo ^ ROL(BCa, 1);

        Eba ^= Da;
        BCa = Eba;
        Ege ^= De;
        BCe = ROL(Ege, 44);
        Eki ^= Di;
        BCi = ROL(Eki, 43);
        Emo ^= Do;
        BCo = ROL(Emo, 21);
        Esu ^= Du;
        BCu = ROL(Esu, 14);
        Aba = BCa ^ ((~BCe) & BCi);
        Aba ^= gpu_KeccakF_RoundConstants[round + 1];
        Abe = BCe ^ ((~BCi) & BCo);
        Abi = BCi ^ ((~BCo) & BCu);
        Abo = BCo ^ ((~BCu) & BCa);
        Abu = BCu ^ ((~BCa) & BCe);

        Ebo ^= Do;
        BCa = ROL(Ebo, 28);
        Egu ^= Du;
        BCe = ROL(Egu, 20);
        Eka ^= Da;
        BCi = ROL(Eka, 3);
        Eme ^= De;
        BCo = ROL(Eme, 45);
        Esi ^= Di;
        BCu = ROL(Esi, 61);
        Aga = BCa ^ ((~BCe) & BCi);
        Age = BCe ^ ((~BCi) & BCo);
        Agi = BCi ^ ((~BCo) & BCu);
        Ago = BCo ^ ((~BCu) & BCa);
        Agu = BCu ^ ((~BCa) & BCe);

        Ebe ^= De;
        BCa = ROL(Ebe, 1);
        Egi ^= Di;
        BCe = ROL(Egi, 6);
        Eko ^= Do;
        BCi = ROL(Eko, 25);
        Emu ^= Du;
        BCo = ROL(Emu, 8);
        Esa ^= Da;
        BCu = ROL(Esa, 18);
        Aka = BCa ^ ((~BCe) & BCi);
        Ake = BCe ^ ((~BCi) & BCo);
        Aki = BCi ^ ((~BCo) & BCu);
        Ako = BCo ^ ((~BCu) & BCa);
        Aku = BCu ^ ((~BCa) & BCe);

        Ebu ^= Du;
        BCa = ROL(Ebu, 27);
        Ega ^= Da;
        BCe = ROL(Ega, 36);
        Eke ^= De;
        BCi = ROL(Eke, 10);
        Emi ^= Di;
        BCo = ROL(Emi, 15);
        Eso ^= Do;
        BCu = ROL(Eso, 56);
        Ama = BCa ^ ((~BCe) & BCi);
        Ame = BCe ^ ((~BCi) & BCo);
        Ami = BCi ^ ((~BCo) & BCu);
        Amo = BCo ^ ((~BCu) & BCa);
        Amu = BCu ^ ((~BCa) & BCe);

        Ebi ^= Di;
        BCa = ROL(Ebi, 62);
        Ego ^= Do;
        BCe = ROL(Ego, 55);
        Eku ^= Du;
        BCi = ROL(Eku, 39);
        Ema ^= Da;
        BCo = ROL(Ema, 41);
        Ese ^= De;
        BCu = ROL(Ese, 2);
        Asa = BCa ^ ((~BCe) & BCi);
        Ase = BCe ^ ((~BCi) & BCo);
        Asi = BCi ^ ((~BCo) & BCu);
        Aso = BCo ^ ((~BCu) & BCa);
        Asu = BCu ^ ((~BCa) & BCe);
    }

    state[0]  = Aba;
    state[1]  = Abe;
    state[2]  = Abi;
    state[3]  = Abo;
    state[4]  = Abu;
    state[5]  = Aga;
    state[6]  = Age;
    state[7]  = Agi;
    state[8]  = Ago;
    state[9]  = Agu;
    state[10] = Aka;
    state[11] = Ake;
    state[12] = Aki;
    state[13] = Ako;
    state[14] = Aku;
    state[15] = Ama;
    state[16] = Ame;
    state[17] = Ami;
    state[18] = Amo;
    state[19] = Amu;
    state[20] = Asa;
    state[21] = Ase;
    state[22] = Asi;
    state[23] = Aso;
    state[24] = Asu;
}

static __device__ void keccak_init(uint64_t s[25])
{
    for(unsigned int i = 0; i < 25; i++)
        s[i] = 0;
}

static __device__ unsigned int
    keccak_absorb(uint64_t s[25], unsigned int pos, unsigned int r, const uint8_t* in, size_t inlen)
{
    unsigned int i;
    while(pos + inlen >= r)
    {
        for(i = pos; i < r; i++)
            s[i / 8] ^= (uint64_t)*in++ << 8 * (i % 8);
        inlen -= r - pos;
        KeccakF1600_StatePermute(s);
        pos = 0;
    }
    for(i = pos; i < pos + (unsigned int)inlen; i++)
        s[i / 8] ^= (uint64_t)*in++ << 8 * (i % 8);
    return i;
}

static __device__ void keccak_finalize(uint64_t s[25], unsigned int pos, unsigned int r, uint8_t p)
{
    s[pos / 8] ^= (uint64_t)p << 8 * (pos % 8);
    s[r / 8 - 1] ^= 1ULL << 63;
}

static __device__ unsigned int
    keccak_squeeze(uint8_t* out, size_t outlen, uint64_t s[25], unsigned int pos, unsigned int r)
{
    unsigned int i;
    while(outlen)
    {
        if(pos == r)
        {
            KeccakF1600_StatePermute(s);
            pos = 0;
        }
        for(i = pos; i < r && i < pos + (unsigned int)outlen; i++)
            *out++ = (uint8_t)(s[i / 8] >> 8 * (i % 8));
        outlen -= i - pos;
        pos = i;
    }
    return pos;
}

static __device__ void
    keccak_absorb_once(uint64_t s[25], unsigned int r, const uint8_t* in, size_t inlen, uint8_t p)
{
    unsigned int i;
    for(i = 0; i < 25; ++i)
        s[i] = 0;
    while(inlen >= r)
    {
        for(i = 0; i < r / 8; ++i)
            s[i] ^= gpu_load64(in + 8 * i);
        KeccakF1600_StatePermute(s);
        inlen -= r;
        in += r;
    }
    for(i = 0; i < (unsigned int)inlen; ++i)
        s[i >> 3] ^= (uint64_t)in[i] << (8 * (i & 7));
    s[inlen >> 3] ^= (uint64_t)p << (8 * (inlen & 7));
    s[(r - 1) >> 3] ^= 1ULL << 63;
}

static __device__ void
    keccak_squeezeblocks(uint8_t* out, size_t nblocks, uint64_t s[25], unsigned int r)
{
    unsigned int i;
    while(nblocks > 0)
    {
        KeccakF1600_StatePermute(s);
        for(i = 0; i < (r >> 3); i++)
            gpu_store64(out + 8 * i, s[i]);
        out += r;
        nblocks--;
    }
}

/* SHAKE128 */
static __device__ void shake128_init(keccak_state* state)
{
    keccak_init(state->s);
    state->pos = 0;
}

static __device__ void shake128_absorb(keccak_state* state, const uint8_t* in, size_t inlen)
{
    state->pos = keccak_absorb(state->s, state->pos, SHAKE128_RATE, in, inlen);
}

static __device__ void shake128_finalize(keccak_state* state)
{
    keccak_finalize(state->s, state->pos, SHAKE128_RATE, 0x1F);
    state->pos = SHAKE128_RATE;
}

static __device__ void shake128_squeeze(uint8_t* out, size_t outlen, keccak_state* state)
{
    state->pos = keccak_squeeze(out, outlen, state->s, state->pos, SHAKE128_RATE);
}

static __device__ void shake128_absorb_once(keccak_state* state, const uint8_t* in, size_t inlen)
{
    keccak_absorb_once(state->s, SHAKE128_RATE, in, inlen, 0x1F);
    state->pos = SHAKE128_RATE;
}

static __device__ void shake128_squeezeblocks(uint8_t* output, size_t nblocks, keccak_state* state)
{
    keccak_squeezeblocks(output, nblocks, state->s, SHAKE128_RATE);
}

/* SHAKE256 */
static __device__ void shake256_init(keccak_state* state)
{
    keccak_init(state->s);
    state->pos = 0;
}

static __device__ void shake256_absorb(keccak_state* state, const uint8_t* in, size_t inlen)
{
    state->pos = keccak_absorb(state->s, state->pos, SHAKE256_RATE, in, inlen);
}

static __device__ void shake256_finalize(keccak_state* state)
{
    keccak_finalize(state->s, state->pos, SHAKE256_RATE, 0x1F);
    state->pos = SHAKE256_RATE;
}

static __device__ void shake256_squeeze(uint8_t* out, size_t outlen, keccak_state* state)
{
    state->pos = keccak_squeeze(out, outlen, state->s, state->pos, SHAKE256_RATE);
}

static __device__ void shake256_absorb_once(keccak_state* state, const uint8_t* input, size_t inlen)
{
    keccak_absorb_once(state->s, SHAKE256_RATE, input, inlen, 0x1F);
    state->pos = SHAKE256_RATE;
}

static __device__ void shake256_squeezeblocks(uint8_t* output, size_t nblocks, keccak_state* state)
{
    keccak_squeezeblocks(output, nblocks, state->s, SHAKE256_RATE);
}

static __device__ __noinline__ void
    shake128(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen)
{
    keccak_state state;
    shake128_absorb_once(&state, in, inlen);
    size_t nblocks = outlen / SHAKE128_RATE;
    shake128_squeezeblocks(out, nblocks, &state);
    outlen -= nblocks * SHAKE128_RATE;
    out += nblocks * SHAKE128_RATE;
    shake128_squeeze(out, outlen, &state);
}

static __device__ __noinline__ void
    shake256(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen)
{
    keccak_state state;
    shake256_absorb_once(&state, in, inlen);
    size_t nblocks = outlen / SHAKE256_RATE;
    shake256_squeezeblocks(out, nblocks, &state);
    outlen -= nblocks * SHAKE256_RATE;
    out += nblocks * SHAKE256_RATE;
    shake256_squeeze(out, outlen, &state);
}

/* SHA3 */
static __device__ __noinline__ void sha3_256(uint8_t* output, const uint8_t* input, size_t inlen)
{
    uint64_t s[25];
    keccak_absorb_once(s, SHA3_256_RATE, input, inlen, 0x06);
    KeccakF1600_StatePermute(s);
    for(size_t i = 0; i < 4; i++)
        gpu_store64(output + 8 * i, s[i]);
}

static __device__ __noinline__ void sha3_512(uint8_t* output, const uint8_t* input, size_t inlen)
{
    uint64_t s[25];
    keccak_absorb_once(s, SHA3_512_RATE, input, inlen, 0x06);
    KeccakF1600_StatePermute(s);
    for(size_t i = 0; i < 8; i++)
        gpu_store64(output + 8 * i, s[i]);
}

#endif /* FIPS202_HPP */
