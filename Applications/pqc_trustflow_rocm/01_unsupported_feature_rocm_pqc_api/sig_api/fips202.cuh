#include "hip/hip_runtime.h"
#ifndef FIPS202_CUH
#define FIPS202_CUH

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_512_RATE 72

typedef struct {
  uint64_t s[25];
  unsigned int pos;
} keccak_state;

#define NROUNDS 24
#define ROL(a, offset) ((a << offset) ^ (a >> (64-offset)))

static __device__ __forceinline__ uint64_t load64(const uint8_t x[8]) {
  uint64_t r;
  memcpy(&r, x, 8);
  return r;
}

static __device__ __forceinline__ void store64(uint8_t x[8], uint64_t u) {
  memcpy(x, &u, 8);
}

__constant__ uint64_t gpu_KeccakF_RoundConstants[NROUNDS] = {
  (uint64_t)0x0000000000000001ULL,
  (uint64_t)0x0000000000008082ULL,
  (uint64_t)0x800000000000808aULL,
  (uint64_t)0x8000000080008000ULL,
  (uint64_t)0x000000000000808bULL,
  (uint64_t)0x0000000080000001ULL,
  (uint64_t)0x8000000080008081ULL,
  (uint64_t)0x8000000000008009ULL,
  (uint64_t)0x000000000000008aULL,
  (uint64_t)0x0000000000000088ULL,
  (uint64_t)0x0000000080008009ULL,
  (uint64_t)0x000000008000000aULL,
  (uint64_t)0x000000008000808bULL,
  (uint64_t)0x800000000000008bULL,
  (uint64_t)0x8000000000008089ULL,
  (uint64_t)0x8000000000008003ULL,
  (uint64_t)0x8000000000008002ULL,
  (uint64_t)0x8000000000000080ULL,
  (uint64_t)0x000000000000800aULL,
  (uint64_t)0x800000008000000aULL,
  (uint64_t)0x8000000080008081ULL,
  (uint64_t)0x8000000000008080ULL,
  (uint64_t)0x0000000080000001ULL,
  (uint64_t)0x8000000080008008ULL
};

/* GPU-optimized Keccak-f[1600] — compact single-round loop
 * Based on Kyber batch_keccak.cu pattern:
 * - 24-element cycle for Rho+Pi in-place (eliminates B[25] temp array)
 * - Row-by-row Chi with 5 temporaries
 * - ~60 registers vs ~120 for 2-round unrolled version
 * - Better occupancy on GPU → higher hash throughput
 */
static __device__ __noinline__ void KeccakF1600_StatePermute(uint64_t state[25])
{
    uint64_t Cx[5], Dx[5];

    for (int round = 0; round < NROUNDS; round++) {
        /* Theta */
        Cx[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
        Cx[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
        Cx[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
        Cx[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
        Cx[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

        Dx[0] = Cx[4] ^ ROL(Cx[1], 1);
        Dx[1] = Cx[0] ^ ROL(Cx[2], 1);
        Dx[2] = Cx[1] ^ ROL(Cx[3], 1);
        Dx[3] = Cx[2] ^ ROL(Cx[4], 1);
        Dx[4] = Cx[3] ^ ROL(Cx[0], 1);

        state[ 0] ^= Dx[0]; state[ 5] ^= Dx[0]; state[10] ^= Dx[0]; state[15] ^= Dx[0]; state[20] ^= Dx[0];
        state[ 1] ^= Dx[1]; state[ 6] ^= Dx[1]; state[11] ^= Dx[1]; state[16] ^= Dx[1]; state[21] ^= Dx[1];
        state[ 2] ^= Dx[2]; state[ 7] ^= Dx[2]; state[12] ^= Dx[2]; state[17] ^= Dx[2]; state[22] ^= Dx[2];
        state[ 3] ^= Dx[3]; state[ 8] ^= Dx[3]; state[13] ^= Dx[3]; state[18] ^= Dx[3]; state[23] ^= Dx[3];
        state[ 4] ^= Dx[4]; state[ 9] ^= Dx[4]; state[14] ^= Dx[4]; state[19] ^= Dx[4]; state[24] ^= Dx[4];

        /* Rho + Pi in-place via 24-element cycle (state[0] is fixed point) */
        {
            uint64_t tmp = ROL(state[1], 1);
            state[ 1] = ROL(state[ 6], 44);
            state[ 6] = ROL(state[ 9], 20);
            state[ 9] = ROL(state[22], 61);
            state[22] = ROL(state[14], 39);
            state[14] = ROL(state[20], 18);
            state[20] = ROL(state[ 2], 62);
            state[ 2] = ROL(state[12], 43);
            state[12] = ROL(state[13], 25);
            state[13] = ROL(state[19],  8);
            state[19] = ROL(state[23], 56);
            state[23] = ROL(state[15], 41);
            state[15] = ROL(state[ 4], 27);
            state[ 4] = ROL(state[24], 14);
            state[24] = ROL(state[21],  2);
            state[21] = ROL(state[ 8], 55);
            state[ 8] = ROL(state[16], 45);
            state[16] = ROL(state[ 5], 36);
            state[ 5] = ROL(state[ 3], 28);
            state[ 3] = ROL(state[18], 21);
            state[18] = ROL(state[17], 15);
            state[17] = ROL(state[11], 10);
            state[11] = ROL(state[ 7],  6);
            state[ 7] = ROL(state[10],  3);
            state[10] = tmp;
        }

        /* Chi — row by row with 5 temporaries */
        {
            uint64_t t0, t1, t2, t3, t4;

            t0=state[0]; t1=state[1]; t2=state[2]; t3=state[3]; t4=state[4];
            state[0]=t0^((~t1)&t2); state[1]=t1^((~t2)&t3);
            state[2]=t2^((~t3)&t4); state[3]=t3^((~t4)&t0);
            state[4]=t4^((~t0)&t1);

            t0=state[5]; t1=state[6]; t2=state[7]; t3=state[8]; t4=state[9];
            state[5]=t0^((~t1)&t2); state[6]=t1^((~t2)&t3);
            state[7]=t2^((~t3)&t4); state[8]=t3^((~t4)&t0);
            state[9]=t4^((~t0)&t1);

            t0=state[10]; t1=state[11]; t2=state[12]; t3=state[13]; t4=state[14];
            state[10]=t0^((~t1)&t2); state[11]=t1^((~t2)&t3);
            state[12]=t2^((~t3)&t4); state[13]=t3^((~t4)&t0);
            state[14]=t4^((~t0)&t1);

            t0=state[15]; t1=state[16]; t2=state[17]; t3=state[18]; t4=state[19];
            state[15]=t0^((~t1)&t2); state[16]=t1^((~t2)&t3);
            state[17]=t2^((~t3)&t4); state[18]=t3^((~t4)&t0);
            state[19]=t4^((~t0)&t1);

            t0=state[20]; t1=state[21]; t2=state[22]; t3=state[23]; t4=state[24];
            state[20]=t0^((~t1)&t2); state[21]=t1^((~t2)&t3);
            state[22]=t2^((~t3)&t4); state[23]=t3^((~t4)&t0);
            state[24]=t4^((~t0)&t1);
        }

        /* Iota */
        state[0] ^= gpu_KeccakF_RoundConstants[round];
    }
}

static __device__ void keccak_init(uint64_t s[25]) {
  unsigned int i;
  for(i=0;i<25;i++) s[i] = 0;
}

/* Word-aligned absorb: uses 64-bit loads when pos is 8-byte aligned
 * All SHAKE/SHA3 rates are multiples of 8, and ML-DSA verify keeps
 * pos 8-byte aligned throughout (mu=64B, w1_pack=128B). */
static __device__ __noinline__ unsigned int keccak_absorb(uint64_t s[25],
                                  unsigned int pos,
                                  unsigned int r,
                                  const uint8_t *in,
                                  size_t inlen)
{
  unsigned int i;
  while(pos+inlen >= r) {
    if (!(pos & 7)) {
      /* Fast path: 64-bit word loads (8x fewer iterations) */
      for(i = pos >> 3; i < r >> 3; i++, in += 8)
        s[i] ^= load64(in);
    } else {
      for(i=pos;i<r;i++)
        s[i/8] ^= (uint64_t)*in++ << 8*(i%8);
    }
    inlen -= r-pos;
    KeccakF1600_StatePermute(s);
    pos = 0;
  }
  /* Remaining partial block */
  if (!(pos & 7) && !(inlen & 7) && inlen > 0) {
    for(i = pos >> 3; i < (pos + (unsigned int)inlen) >> 3; i++, in += 8)
      s[i] ^= load64(in);
    return pos + (unsigned int)inlen;
  }
  for(i=pos;i<pos+(unsigned int)inlen;i++)
    s[i/8] ^= (uint64_t)*in++ << 8*(i%8);
  return i;
}

static __device__ void keccak_finalize(uint64_t s[25], unsigned int pos, unsigned int r, uint8_t p) {
  s[pos/8] ^= (uint64_t)p << 8*(pos%8);
  s[r/8-1] ^= 1ULL << 63;
}

/* Word-aligned squeeze: uses 64-bit stores when pos and outlen allow */
static __device__ __noinline__ unsigned int keccak_squeeze(uint8_t *out,
                                   size_t outlen,
                                   uint64_t s[25],
                                   unsigned int pos,
                                   unsigned int r)
{
  unsigned int i;
  while(outlen) {
    if(pos == r) {
      KeccakF1600_StatePermute(s);
      pos = 0;
    }
    if (!(pos & 7) && outlen >= 8) {
      /* Fast path: word-aligned squeeze */
      unsigned int end = (pos + (unsigned int)outlen < r) ? (pos + (unsigned int)outlen) : r;
      for(i = pos >> 3; i < end >> 3; i++, out += 8)
        store64(out, s[i]);
      unsigned int extracted = (i << 3) - pos;
      outlen -= extracted;
      pos += extracted;
    } else {
      for(i=pos;i < r && i < pos+outlen; i++)
        *out++ = s[i/8] >> 8*(i%8);
      outlen -= i-pos;
      pos = i;
    }
  }
  return pos;
}

static __device__ __noinline__ void keccak_absorb_once(uint64_t s[25],
                               unsigned int r,
                               const uint8_t *in,
                               size_t inlen,
                               uint8_t p)
{
  unsigned int i;
  for(i=0;i<25;i++) s[i] = 0;
  while(inlen >= r) {
    for(i=0;i<r/8;i++)
      s[i] ^= load64(in+8*i);
    in += r;
    inlen -= r;
    KeccakF1600_StatePermute(s);
  }
  for(i=0;i<inlen;i++)
    s[i/8] ^= (uint64_t)in[i] << 8*(i%8);
  s[i/8] ^= (uint64_t)p << 8*(i%8);
  s[(r-1)/8] ^= 1ULL << 63;
}

static __device__ void keccak_squeezeblocks(uint8_t *out, size_t nblocks, uint64_t s[25], unsigned int r) {
  unsigned int i;
  while(nblocks) {
    KeccakF1600_StatePermute(s);
    for(i=0;i<r/8;i++)
      store64(out+8*i, s[i]);
    out += r;
    nblocks -= 1;
  }
}

/* SHAKE128 */
static __device__ void shake128_init(keccak_state *state) {
  keccak_init(state->s); state->pos = 0;
}

static __device__ void shake128_absorb(keccak_state *state, const uint8_t *in, size_t inlen) {
  state->pos = keccak_absorb(state->s, state->pos, SHAKE128_RATE, in, inlen);
}

static __device__ void shake128_finalize(keccak_state *state) {
  keccak_finalize(state->s, state->pos, SHAKE128_RATE, 0x1F);
  state->pos = SHAKE128_RATE;
}

static __device__ void shake128_squeeze(uint8_t *out, size_t outlen, keccak_state *state) {
  state->pos = keccak_squeeze(out, outlen, state->s, state->pos, SHAKE128_RATE);
}

static __device__ void shake128_absorb_once(keccak_state *state, const uint8_t *in, size_t inlen) {
  keccak_absorb_once(state->s, SHAKE128_RATE, in, inlen, 0x1F);
  state->pos = SHAKE128_RATE;
}

static __device__ void shake128_squeezeblocks(uint8_t *out, size_t nblocks, keccak_state *state) {
  keccak_squeezeblocks(out, nblocks, state->s, SHAKE128_RATE);
}

/* SHAKE256 */
static __device__ void shake256_init(keccak_state *state) {
  keccak_init(state->s); state->pos = 0;
}

static __device__ void shake256_absorb(keccak_state *state, const uint8_t *in, size_t inlen) {
  state->pos = keccak_absorb(state->s, state->pos, SHAKE256_RATE, in, inlen);
}

static __device__ void shake256_finalize(keccak_state *state) {
  keccak_finalize(state->s, state->pos, SHAKE256_RATE, 0x1F);
  state->pos = SHAKE256_RATE;
}

static __device__ void shake256_squeeze(uint8_t *out, size_t outlen, keccak_state *state) {
  state->pos = keccak_squeeze(out, outlen, state->s, state->pos, SHAKE256_RATE);
}

static __device__ void shake256_absorb_once(keccak_state *state, const uint8_t *in, size_t inlen) {
  keccak_absorb_once(state->s, SHAKE256_RATE, in, inlen, 0x1F);
  state->pos = SHAKE256_RATE;
}

static __device__ void shake256_squeezeblocks(uint8_t *out, size_t nblocks, keccak_state *state) {
  keccak_squeezeblocks(out, nblocks, state->s, SHAKE256_RATE);
}

/* Non-incremental API */
static __device__ __noinline__ void shake128(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen) {
  size_t nblocks;
  keccak_state state;
  shake128_absorb_once(&state, in, inlen);
  nblocks = outlen/SHAKE128_RATE;
  shake128_squeezeblocks(out, nblocks, &state);
  outlen -= nblocks*SHAKE128_RATE;
  out += nblocks*SHAKE128_RATE;
  shake128_squeeze(out, outlen, &state);
}

static __device__ __noinline__ void shake256(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen) {
  size_t nblocks;
  keccak_state state;
  shake256_absorb_once(&state, in, inlen);
  nblocks = outlen/SHAKE256_RATE;
  shake256_squeezeblocks(out, nblocks, &state);
  outlen -= nblocks*SHAKE256_RATE;
  out += nblocks*SHAKE256_RATE;
  shake256_squeeze(out, outlen, &state);
}

static __device__ void sha3_256(uint8_t h[32], const uint8_t *in, size_t inlen) {
  unsigned int i;
  uint64_t s[25];
  keccak_absorb_once(s, SHA3_256_RATE, in, inlen, 0x06);
  KeccakF1600_StatePermute(s);
  for(i=0;i<4;i++) store64(h+8*i,s[i]);
}

static __device__ void sha3_512(uint8_t h[64], const uint8_t *in, size_t inlen) {
  unsigned int i;
  uint64_t s[25];
  keccak_absorb_once(s, SHA3_512_RATE, in, inlen, 0x06);
  KeccakF1600_StatePermute(s);
  for(i=0;i<8;i++) store64(h+8*i,s[i]);
}

#endif
