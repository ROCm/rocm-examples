/*
 * cbd.cuh — 中心二项分布 (CBD) 噪声采样
 *
 * 支持 eta = 1, 2, 3, 4, 8 (Kyber 和 Aigis-enc 共用)
 *
 * 各算法用到的 eta 值:
 *   Kyber: ETA1 = 2 (Kyber768/1024) 或 3 (Kyber512); ETA2 = 2
 *   Aigis-enc: ETA_S=1/2/3/4, ETA_E_KG=4/4/4/8, ETA_E_ENC=4/4/4/8, ETA_E2
 *
 * 实现方法:
 *   对 eta 个比特对的求和: sum_{i=0}^{eta-1} (bit_a_i - bit_b_i)
 *   其中 a 和 b 是两组独立的随机比特
 */

#ifndef CBD_CUH
#define CBD_CUH

#include <stdint.h>
#include "params.h"
#include "fips202.cuh"

/* ================================================================
 *  底层 CBD 函数 (根据 eta 分发)
 * ================================================================ */

/* eta=1: 每字节产生 4 个系数 (每两位) */
static __device__ void cbd1(int16_t *r, const uint8_t *buf, unsigned int len)
{
    unsigned int pos = 0, i;
    for (i = 0; i + 3 < len * 4 && pos < (unsigned)PARAM_N; i += 4) {
        uint8_t b = buf[i / 4];
        r[pos++] = (int16_t)(((b >> 0) & 1) - ((b >> 1) & 1));
        r[pos++] = (int16_t)(((b >> 2) & 1) - ((b >> 3) & 1));
        r[pos++] = (int16_t)(((b >> 4) & 1) - ((b >> 5) & 1));
        r[pos++] = (int16_t)(((b >> 6) & 1) - ((b >> 7) & 1));
    }
}

/* eta=2: 每 4 字节产生 8 个系数 */
static __device__ void cbd2(int16_t *r, const uint8_t *buf)
{
    /* 处理 256 个系数, 需 256*4/8=128 字节 */
    for (unsigned int i = 0; i < PARAM_N / 8; i++) {
        uint32_t t  = (uint32_t)buf[4*i+0] | ((uint32_t)buf[4*i+1] << 8) |
                      ((uint32_t)buf[4*i+2] << 16) | ((uint32_t)buf[4*i+3] << 24);
        uint32_t d = t & 0x55555555;
        d += (t >> 1) & 0x55555555;
        r[8*i+0] = (int16_t)(((d >>  0) & 0x3) - ((d >>  2) & 0x3));
        r[8*i+1] = (int16_t)(((d >>  4) & 0x3) - ((d >>  6) & 0x3));
        r[8*i+2] = (int16_t)(((d >>  8) & 0x3) - ((d >> 10) & 0x3));
        r[8*i+3] = (int16_t)(((d >> 12) & 0x3) - ((d >> 14) & 0x3));
        r[8*i+4] = (int16_t)(((d >> 16) & 0x3) - ((d >> 18) & 0x3));
        r[8*i+5] = (int16_t)(((d >> 20) & 0x3) - ((d >> 22) & 0x3));
        r[8*i+6] = (int16_t)(((d >> 24) & 0x3) - ((d >> 26) & 0x3));
        r[8*i+7] = (int16_t)(((d >> 28) & 0x3) - ((d >> 30) & 0x3));
    }
}

/* eta=3: 每 3 字节产生 4 个系数 */
static __device__ void cbd3(int16_t *r, const uint8_t *buf)
{
    /* 需 256 * 3 / 4 * ... 实际需 192 字节 */
    for (unsigned int i = 0; i < PARAM_N / 4; i++) {
        uint32_t a, b;
        uint32_t t = (uint32_t)buf[3*i+0] | ((uint32_t)buf[3*i+1] << 8) | ((uint32_t)buf[3*i+2] << 16);
        a  =  t & 0x249249;
        a += (t >> 1) & 0x249249;
        a += (t >> 2) & 0x249249;
        b  = (t >> 3) & 0x249249;
        b += (t >> 4) & 0x249249;
        b += (t >> 5) & 0x249249;
        r[4*i]   = (int16_t)(((a >>  0) & 0x7) - ((b >>  0) & 0x7));
        r[4*i+1] = (int16_t)(((a >>  6) & 0x7) - ((b >>  6) & 0x7));
        r[4*i+2] = (int16_t)(((a >> 12) & 0x7) - ((b >> 12) & 0x7));
        r[4*i+3] = (int16_t)(((a >> 18) & 0x7) - ((b >> 18) & 0x7));
    }
}

/* eta=4: 每 2 字节产生 2 个系数 */
static __device__ void cbd4(int16_t *r, const uint8_t *buf)
{
    for (unsigned int i = 0; i < PARAM_N / 2; i++) {
        uint32_t t = (uint32_t)buf[2*i] | ((uint32_t)buf[2*i+1] << 8);
        uint32_t a = t & 0x1111;
        a += (t >> 1) & 0x1111;
        a += (t >> 2) & 0x1111;
        a += (t >> 3) & 0x1111;
        uint32_t b = (t >>  8) & 0x1111;
        b += (t >>  9) & 0x1111;
        b += (t >> 10) & 0x1111;
        b += (t >> 11) & 0x1111;
        r[2*i]   = (int16_t)((a & 0xF) - (b & 0xF));
        r[2*i+1] = (int16_t)(((a >> 4) & 0xF) - ((b >> 4) & 0xF));
    }
}

/* eta=8: 每字节产生 1 个系数 (popcount 差) */
static __device__ void cbd8(int16_t *r, const uint8_t *buf)
{
    for (unsigned int i = 0; i < PARAM_N; i++) {
        uint8_t a = buf[2*i];
        uint8_t b = buf[2*i+1];
        r[i] = (int16_t)(__popc((unsigned)a) - __popc((unsigned)b));
    }
}

/* ================================================================
 *  SHAKE256 PRF: out = SHAKE256(seed || nonce, outlen)
 * ================================================================ */
static __device__ __noinline__ void prf_shake256(uint8_t *out, size_t outlen,
    const uint8_t *seed, uint8_t nonce)
{
    uint64_t s[25];
    for (unsigned int i = 0; i < 25; i++) s[i] = 0;
    for (unsigned int i = 0; i < PARAM_SYMBYTES; i++)
        s[i >> 3] ^= (uint64_t)seed[i] << (8 * (i & 7));
    s[PARAM_SYMBYTES >> 3] ^= (uint64_t)nonce << (8 * (PARAM_SYMBYTES & 7));
    s[(PARAM_SYMBYTES + 1) >> 3] ^= (uint64_t)0x1F << (8 * ((PARAM_SYMBYTES + 1) & 7));
    s[(SHAKE256_RATE - 1) >> 3] ^= 1ULL << 63;

    size_t nblocks = outlen / SHAKE256_RATE;
    keccak_squeezeblocks(out, nblocks, s, SHAKE256_RATE);
    outlen -= nblocks * SHAKE256_RATE;
    out += nblocks * SHAKE256_RATE;
    if (outlen) {
        KeccakF1600_StatePermute(s);
        for (size_t i = 0; i < outlen; i++)
            out[i] = (uint8_t)(s[i >> 3] >> (8 * (i & 7)));
    }
}

/* ================================================================
 *  统一噪声采样接口
 *  getnoise_eta(r, seed, nonce, eta):
 *    使用 SHAKE256(seed||nonce) 生成 CBD(eta) 多项式
 * ================================================================ */
#ifndef KEM_DIRECT_CBD
#define KEM_DIRECT_CBD 1
#endif

#if KEM_DIRECT_CBD
typedef struct {
    uint64_t s[25];
    unsigned int pos;
} prf_reader;

static __device__ __forceinline__ void prf_reader_init(prf_reader *rd,
    const uint8_t *seed, uint8_t nonce)
{
    for (unsigned int i = 0; i < 25; i++) rd->s[i] = 0;
    for (unsigned int i = 0; i < PARAM_SYMBYTES; i++)
        rd->s[i >> 3] ^= (uint64_t)seed[i] << (8 * (i & 7));
    rd->s[PARAM_SYMBYTES >> 3] ^= (uint64_t)nonce << (8 * (PARAM_SYMBYTES & 7));
    rd->s[(PARAM_SYMBYTES + 1) >> 3] ^= (uint64_t)0x1F << (8 * ((PARAM_SYMBYTES + 1) & 7));
    rd->s[(SHAKE256_RATE - 1) >> 3] ^= 1ULL << 63;
    rd->pos = SHAKE256_RATE;
}

static __device__ __forceinline__ uint8_t prf_reader_u8(prf_reader *rd)
{
    if (rd->pos == SHAKE256_RATE) {
        KeccakF1600_StatePermute(rd->s);
        rd->pos = 0;
    }
    uint8_t v = (uint8_t)(rd->s[rd->pos >> 3] >> (8 * (rd->pos & 7)));
    rd->pos++;
    return v;
}

static __device__ __forceinline__ uint16_t prf_reader_u16(prf_reader *rd)
{
    uint16_t b0 = prf_reader_u8(rd);
    uint16_t b1 = prf_reader_u8(rd);
    return (uint16_t)(b0 | (b1 << 8));
}

static __device__ __forceinline__ uint32_t prf_reader_u24(prf_reader *rd)
{
    uint32_t b0 = prf_reader_u8(rd);
    uint32_t b1 = prf_reader_u8(rd);
    uint32_t b2 = prf_reader_u8(rd);
    return b0 | (b1 << 8) | (b2 << 16);
}

static __device__ __forceinline__ uint32_t prf_reader_u32(prf_reader *rd)
{
    uint32_t b0 = prf_reader_u8(rd);
    uint32_t b1 = prf_reader_u8(rd);
    uint32_t b2 = prf_reader_u8(rd);
    uint32_t b3 = prf_reader_u8(rd);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static __device__ __noinline__ void getnoise_eta1(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    prf_reader rd;
    prf_reader_init(&rd, seed, nonce);
    for (unsigned int i = 0; i < PARAM_N / 4; i++) {
        uint8_t b = prf_reader_u8(&rd);
        r[4*i+0] = (int16_t)(((b >> 0) & 1) - ((b >> 1) & 1));
        r[4*i+1] = (int16_t)(((b >> 2) & 1) - ((b >> 3) & 1));
        r[4*i+2] = (int16_t)(((b >> 4) & 1) - ((b >> 5) & 1));
        r[4*i+3] = (int16_t)(((b >> 6) & 1) - ((b >> 7) & 1));
    }
}

static __device__ __noinline__ void getnoise_eta2(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    prf_reader rd;
    prf_reader_init(&rd, seed, nonce);
    for (unsigned int i = 0; i < PARAM_N / 8; i++) {
        uint32_t t = prf_reader_u32(&rd);
        uint32_t d = t & 0x55555555;
        d += (t >> 1) & 0x55555555;
        r[8*i+0] = (int16_t)(((d >>  0) & 0x3) - ((d >>  2) & 0x3));
        r[8*i+1] = (int16_t)(((d >>  4) & 0x3) - ((d >>  6) & 0x3));
        r[8*i+2] = (int16_t)(((d >>  8) & 0x3) - ((d >> 10) & 0x3));
        r[8*i+3] = (int16_t)(((d >> 12) & 0x3) - ((d >> 14) & 0x3));
        r[8*i+4] = (int16_t)(((d >> 16) & 0x3) - ((d >> 18) & 0x3));
        r[8*i+5] = (int16_t)(((d >> 20) & 0x3) - ((d >> 22) & 0x3));
        r[8*i+6] = (int16_t)(((d >> 24) & 0x3) - ((d >> 26) & 0x3));
        r[8*i+7] = (int16_t)(((d >> 28) & 0x3) - ((d >> 30) & 0x3));
    }
}

static __device__ __noinline__ void getnoise_eta3(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    prf_reader rd;
    prf_reader_init(&rd, seed, nonce);
    for (unsigned int i = 0; i < PARAM_N / 4; i++) {
        uint32_t t = prf_reader_u24(&rd);
        uint32_t a =  t & 0x249249;
        a += (t >> 1) & 0x249249;
        a += (t >> 2) & 0x249249;
        uint32_t b = (t >> 3) & 0x249249;
        b += (t >> 4) & 0x249249;
        b += (t >> 5) & 0x249249;
        r[4*i+0] = (int16_t)(((a >>  0) & 0x7) - ((b >>  0) & 0x7));
        r[4*i+1] = (int16_t)(((a >>  6) & 0x7) - ((b >>  6) & 0x7));
        r[4*i+2] = (int16_t)(((a >> 12) & 0x7) - ((b >> 12) & 0x7));
        r[4*i+3] = (int16_t)(((a >> 18) & 0x7) - ((b >> 18) & 0x7));
    }
}

static __device__ __noinline__ void getnoise_eta4(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    prf_reader rd;
    prf_reader_init(&rd, seed, nonce);
    for (unsigned int i = 0; i < PARAM_N / 2; i++) {
        uint32_t t = prf_reader_u16(&rd);
        uint32_t a = t & 0x1111;
        a += (t >> 1) & 0x1111;
        a += (t >> 2) & 0x1111;
        a += (t >> 3) & 0x1111;
        uint32_t b = (t >>  8) & 0x1111;
        b += (t >>  9) & 0x1111;
        b += (t >> 10) & 0x1111;
        b += (t >> 11) & 0x1111;
        r[2*i+0] = (int16_t)((a & 0xF) - (b & 0xF));
        r[2*i+1] = (int16_t)(((a >> 4) & 0xF) - ((b >> 4) & 0xF));
    }
}

static __device__ __noinline__ void getnoise_eta8(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    prf_reader rd;
    prf_reader_init(&rd, seed, nonce);
    for (unsigned int i = 0; i < PARAM_N; i++) {
        uint8_t a = prf_reader_u8(&rd);
        uint8_t b = prf_reader_u8(&rd);
        r[i] = (int16_t)(__popc((unsigned)a) - __popc((unsigned)b));
    }
}

#else

static __device__ __noinline__ void getnoise_eta1(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    uint8_t buf[1 * 64];
    prf_shake256(buf, sizeof(buf), seed, nonce);
    cbd1(r, buf, (unsigned int)sizeof(buf));
}

static __device__ __noinline__ void getnoise_eta2(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    uint8_t buf[2 * 64];
    prf_shake256(buf, sizeof(buf), seed, nonce);
    cbd2(r, buf);
}

static __device__ __noinline__ void getnoise_eta3(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    uint8_t buf[3 * 64];
    prf_shake256(buf, sizeof(buf), seed, nonce);
    cbd3(r, buf);
}

static __device__ __noinline__ void getnoise_eta4(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    uint8_t buf[4 * 64];
    prf_shake256(buf, sizeof(buf), seed, nonce);
    cbd4(r, buf);
}

static __device__ __noinline__ void getnoise_eta8(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    uint8_t buf[8 * 64];
    prf_shake256(buf, sizeof(buf), seed, nonce);
    cbd8(r, buf);
}

#endif

#define DISPATCH_GETNOISE_ETA(ETA, R, SEED, NONCE) do { \
    if ((ETA) == 1) getnoise_eta1((R), (SEED), (NONCE)); \
    else if ((ETA) == 2) getnoise_eta2((R), (SEED), (NONCE)); \
    else if ((ETA) == 3) getnoise_eta3((R), (SEED), (NONCE)); \
    else if ((ETA) == 4) getnoise_eta4((R), (SEED), (NONCE)); \
    else if ((ETA) == 8) getnoise_eta8((R), (SEED), (NONCE)); \
} while (0)

/* ================================================================
 *  算法特定噪声采样宏 (使用 params.h 中的 eta 参数)
 * ================================================================ */

/* 密钥生成: 秘密向量 s 的噪声 */
static __device__ void poly_getnoise_s(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    DISPATCH_GETNOISE_ETA(PARAM_ETA_S, r, seed, nonce);
}

/* 密钥生成: 错误向量 e 的噪声 */
static __device__ void poly_getnoise_e_kg(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    DISPATCH_GETNOISE_ETA(PARAM_ETA_E_KG, r, seed, nonce);
}

/* 加密: 随机向量 r 的噪声 */
static __device__ void poly_getnoise_e_enc(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    DISPATCH_GETNOISE_ETA(PARAM_ETA_E_ENC, r, seed, nonce);
}

/* 加密: 标量误差 e2 的噪声 */
static __device__ void poly_getnoise_e2(int16_t *r, const uint8_t *seed, uint8_t nonce)
{
    DISPATCH_GETNOISE_ETA(PARAM_ETA_E2, r, seed, nonce);
}

#undef DISPATCH_GETNOISE_ETA

#endif /* CBD_CUH */
