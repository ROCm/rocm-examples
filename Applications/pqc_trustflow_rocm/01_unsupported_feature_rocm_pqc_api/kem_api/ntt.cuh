/*
 * ntt.cuh — 统一 NTT / INVNTT
 *
 * 两种算法使用完全相同的 Cooley-Tukey 蝶形结构，差异仅在于:
 *   1. Q 值不同 → zeta 表数值不同
 *   2. NTT 级数: Kyber=7级(不完全NTT, 最后用 basemul), Aigis-enc=8级(完全NTT, 逐点乘)
 *
 * Kyber NTT 说明:
 *   Q=3329, Q-1=3328=2^8*13, 故 256 | Q-1
 *   使用 7 级 NTT，结果为 128 对 (a[2i], a[2i+1]) 分别处于
 *   二次扩域 Z_q[x]/(x^2 - ζ_i^2) 中
 *   乘法通过 basemul 完成 (见 poly.cuh)
 *   zetas 表: 128 个元素 (indices 1..127, index 0 未使用)
 *
 * Aigis NTT 说明:
 *   Q=7681, Q-1=7680=2^9*3*5, 故 256 | Q-1
 *   使用完整 8 级 NTT，结果为 256 个线性因子元素
 *   乘法为逐点 Montgomery 乘法
 *   zetas 表: 256 个元素; zetas_inv 表: 256 个元素
 */

#ifndef NTT_CUH
#define NTT_CUH

#include <stdint.h>
#include "params.h"
#include "reduce.cuh"

/* ================================================================
 *  Kyber Zeta 表 (128 个元素, indices 1..127)
 *  来源: CRYSTALS-Kyber reference implementation
 * ================================================================ */
#if ALGORITHM == ALGO_KYBER

__constant__ int16_t ntt_zetas[128] = {
   -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
    -171,   622,  1577,   182,   962, -1202, -1474,  1468,
     573, -1325,   264,   383,  -829,  1458, -1602,  -130,
    -681,  1017,   732,   608, -1542,   411,  -205, -1571,
    1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
     516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
    -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
    -398,   961, -1508,  -725,   448, -1065,   677, -1275,
   -1103,   430,   555,   843, -1251,   871,  1550,   105,
     422,   587,   177,  -235,  -291,  -460,  1574,  1653,
    -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
   -1590,   644,  -872,   349,   418,   329,  -156,   -75,
     817,  1097,   603,   610,  1322, -1285, -1465,   384,
   -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
   -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
    -108,  -308,   996,   991,   958, -1460,  1522,  1628
};

/* ================================================================
 *  Aigis-enc Zeta 表 (256 个元素, 支持完整 8 级 NTT)
 *  Q=7681 时使用的 zetas 和 zetas_inv
 * ================================================================ */
#elif ALGORITHM == ALGO_AIGIS_ENC

__constant__ int16_t ntt_zetas[256] = {
    0,3777,-3182,3625,-3696,-1100,2456,2194,121,-2250,834,-2495,-2319,2876,-1701,1414,
    2816,-2088,-2237,1986,-1599,1993,3706,-2006,-1525,-2557,1296,1483,-2830,3364,617,1921,
    -3689,-1738,3266,-3600,810,1887,-638,-7,-438,-679,-1305,-1760,396,-3174,-3555,-1881,
    3772,-2535,-2440,-2555,1535,-549,3153,2310,-1399,1321,514,-2956,-103,2804,-2043,-1431,
    -1054,1698,-3456,1166,2426,3831,915,-2,-3417,-194,2919,2789,3405,2385,-2113,-2732,
    2175,373,3692,-730,-1756,3135,-2391,660,-1497,2572,-3145,1350,-2224,-3588,-1681,2883,
    -1390,1598,3750,2762,2835,2764,-2233,3816,-1533,1464,-727,1521,1386,-3428,-921,-2743,
    -2160,2649,-859,2579,1532,1919,-486,404,-1056,783,1799,-2665,3480,2133,-3310,-1168,
    -17,3744,2422,2001,1278,929,-1348,-2230,-179,-1242,-2059,-1070,2161,1649,2072,3177,
    -2071,1121,-436,236,715,670,-658,-1476,-2378,2767,3542,-226,1203,1181,-151,-3794,
    1712,-222,2786,-451,-3547,1779,-1151,-434,3568,-3693,3581,-1586,1509,2918,2339,-1407,
    3434,-3550,2340,2891,2998,-3314,3461,-2719,-2247,-2589,1144,1072,1295,-2815,-3770,3450,
    3781,-2258,796,3163,-3208,-589,2963,-124,3214,3334,-3366,-3745,3723,1931,-429,-402,
    -3408,83,-1526,826,-1338,2345,-2303,2515,-642,-1837,-2965,-791,370,293,3312,2083,
    -1689,-777,2070,2262,-893,2386,-188,-1519,-2874,-1404,1012,2130,1441,2532,-3335,-1084,
    -3343,2937,509,-1403,2812,3763,592,2005,3657,2460,-3677,3752,692,1669,2167,-3287
};

__constant__ int16_t ntt_zetas_inv[256] = {
    3287,-2167,-1669,-692,-3752,3677,-2460,-3657,-2005,-592,-3763,-2812,1403,-509,-2937,3343,
    1084,3335,-2532,-1441,-2130,-1012,1404,2874,1519,188,-2386,893,-2262,-2070,777,1689,
    -2083,-3312,-293,-370,791,2965,1837,642,-2515,2303,-2345,1338,-826,1526,-83,3408,
    402,429,-1931,-3723,3745,3366,-3334,-3214,124,-2963,589,3208,-3163,-796,2258,-3781,
    -3450,3770,2815,-1295,-1072,-1144,2589,2247,2719,-3461,3314,-2998,-2891,-2340,3550,-3434,
    1407,-2339,-2918,-1509,1586,-3581,3693,-3568,434,1151,-1779,3547,451,-2786,222,-1712,
    3794,151,-1181,-1203,226,-3542,-2767,2378,1476,658,-670,-715,-236,436,-1121,2071,
    -3177,-2072,-1649,-2161,1070,2059,1242,179,2230,1348,-929,-1278,-2001,-2422,-3744,17,
    1168,3310,-2133,-3480,2665,-1799,-783,1056,-404,486,-1919,-1532,-2579,859,-2649,2160,
    2743,921,3428,-1386,-1521,727,-1464,1533,-3816,2233,-2764,-2835,-2762,-3750,-1598,1390,
    -2883,1681,3588,2224,-1350,3145,-2572,1497,-660,2391,-3135,1756,730,-3692,-373,-2175,
    2732,2113,-2385,-3405,-2789,-2919,194,3417,2,-915,-3831,-2426,-1166,3456,-1698,1054,
    1431,2043,-2804,103,2956,-514,-1321,1399,-2310,-3153,549,-1535,2555,2440,2535,-3772,
    1881,3555,3174,-396,1760,1305,679,438,7,638,-1887,-810,3600,-3266,1738,3689,
    -1921,-617,-3364,2830,-1483,-1296,2557,1525,2006,-3706,-1993,1599,-1986,2237,2088,-2816,
    -1414,1701,-2876,2319,2495,-834,2250,-121,-2194,-2456,1100,3696,-3625,3182,-1905
};

/* Aigis INVNTT 归一化因子: mont_invn = N^{-1} * R mod Q
 * N=256, R=2^16=65536
 * 256^{-1} mod 7681 = 7651 (256*7651 = 1958656 ≡ 1 mod 7681 ✓)
 * mont_invn = 7651 * R mod Q  -- 但这已经是 zetas_inv 最后一项的乘积
 * 实际: Aigis INVNTT 把 N^{-1} 折叠进最后一级蝶形 (level 7, step=128) */
#endif  /* ALGORITHM */

/* ================================================================
 *  串行 NTT (单线程，用于 INDCPA 的设备内调用)
 * ================================================================ */

#if ALGORITHM == ALGO_KYBER

/* Kyber 7 级 NTT
 * 蝶形: t = fqmul(zeta, a[j+len]); a[j+len] = a[j]-t; a[j] = a[j]+t
 * 与参考实现完全一致 */
static __device__ __noinline__ void ntt(int16_t r[256])
{
    unsigned int len, start, j, k;
    int16_t t;

    k = 1;
    for (len = 128; len >= 2; len >>= 1) {
        for (start = 0; start < 256; start = j + len) {
            int16_t zeta = ntt_zetas[k++];
            for (j = start; j < start + len; j++) {
                t = fqmul(zeta, r[j + len]);
                r[j + len] = r[j] - t;
                r[j]       = r[j] + t;
            }
        }
    }
}

/* Kyber 7 级 INVNTT + basemul 归一化
 * 末级乘以 f = 1441 = mont(mont(3303)) = (2^16)^2 * 3303 mod Q / Q mod Q...
 * 实际: 参考实现在最后乘以 f=1441 (= N^{-1} * 2^{32} mod Q in Mont)
 * 这里用 f 把系数从 NTT 域缩放回正常范围 */
static __device__ __noinline__ void invntt(int16_t r[256])
{
    unsigned int start, len, j, k;
    int16_t t;
    const int16_t f = 1441; /* mont(3303) 归一化常数 */

    k = 127;
    for (len = 2; len <= 128; len <<= 1) {
        for (start = 0; start < 256; start = j + len) {
            int16_t zeta = ntt_zetas[k--];
            for (j = start; j < start + len; j++) {
                t = r[j];
                r[j]       = barrett_reduce((int16_t)(t + r[j + len]));
                r[j + len] = fqmul(zeta, (int16_t)(r[j + len] - t));
            }
        }
    }
    for (j = 0; j < 256; j++)
        r[j] = fqmul(r[j], f);
}

/* Kyber basemul: 两个度-1 多项式在 Z_q[x]/(x^2-ζ) 中的乘积
 * r[0] = a[0]*b[0] + a[1]*b[1]*zeta
 * r[1] = a[0]*b[1] + a[1]*b[0]  */
static __device__ __forceinline__ void basemul(int16_t r[2],
    const int16_t a[2], const int16_t b[2], int16_t zeta)
{
    r[0] = fqmul(a[1], b[1]);
    r[0] = fqmul(r[0], zeta);
    r[0] += fqmul(a[0], b[0]);
    r[1] = fqmul(a[0], b[1]);
    r[1] += fqmul(a[1], b[0]);
}

/* Kyber polyvec_basemul_acc: r = sum_j a[j] (*) b[j] (basemul 域内积)
 * 每次处理 4 个系数: [4i,4i+1] 用 +zeta[64+i], [4i+2,4i+3] 用 -zeta[64+i]
 * zeta 表共 128 项，[64..127] 用于最后 NTT 级别和 basemul */
static __device__ __noinline__ void polyvec_basemul_acc(kem_poly *r, const kem_polyvec *a, const kem_polyvec *b)
{
    for (int i = 0; i < PARAM_N / 4; i++) {
        int16_t zeta = ntt_zetas[64 + i];  /* indices 64..127 */
        int16_t acc0, acc1;

        /* 第一对 [4i, 4i+1]: 使用 +zeta */
        acc0 = 0; acc1 = 0;
        for (int j = 0; j < PARAM_K; j++) {
            int16_t tmp[2];
            basemul(tmp, &a->vec[j].coeffs[4*i], &b->vec[j].coeffs[4*i], zeta);
            acc0 += tmp[0];
            acc1 += tmp[1];
        }
        r->coeffs[4*i]   = barrett_reduce(acc0);
        r->coeffs[4*i+1] = barrett_reduce(acc1);

        /* 第二对 [4i+2, 4i+3]: 使用 -zeta */
        int16_t neg_zeta = (int16_t)(-zeta);
        acc0 = 0; acc1 = 0;
        for (int j = 0; j < PARAM_K; j++) {
            int16_t tmp[2];
            basemul(tmp, &a->vec[j].coeffs[4*i+2], &b->vec[j].coeffs[4*i+2], neg_zeta);
            acc0 += tmp[0];
            acc1 += tmp[1];
        }
        r->coeffs[4*i+2] = barrett_reduce(acc0);
        r->coeffs[4*i+3] = barrett_reduce(acc1);
    }
}

#elif ALGORITHM == ALGO_AIGIS_ENC

/* Aigis 8 级 NTT (串行)
 * 使用 ntt_zetas[1..255] (index 0 未使用) */
static __device__ __noinline__ void ntt(int16_t r[256])
{
    int start, j, k, step, level;
    int16_t t;

    k = 1;
    /* level 7: step=128 */
    step = 128;
    for (start = 0; start < 256; start = j + step) {
        int16_t zeta = ntt_zetas[k++];
        for (j = start; j < start + step; ++j) {
            t = fqmul(zeta, r[j + step]);
            r[j + step] = r[j] - t;
            r[j]        = r[j] + t;
        }
    }
    /* level 6: step=64 */
    step = 64;
    for (start = 0; start < 256; start = j + step) {
        int16_t zeta = ntt_zetas[k++];
        for (j = start; j < start + step; ++j) {
            t = fqmul(zeta, r[j + step]);
            r[j + step] = barrett_reduce(r[j] - t);
            r[j]        = barrett_reduce(r[j] + t);
        }
    }
    /* levels 5,4 */
    for (level = 5; level >= 4; level--) {
        step = (1 << level);
        for (start = 0; start < 256; start = j + step) {
            int16_t zeta = ntt_zetas[k++];
            for (j = start; j < start + step; ++j) {
                t = fqmul(zeta, r[j + step]);
                r[j + step] = r[j] - t;
                r[j]        = r[j] + t;
            }
        }
    }
    /* level 3: step=8 */
    step = 8;
    for (start = 0; start < 256; start = j + step) {
        int16_t zeta = ntt_zetas[k++];
        for (j = start; j < start + step; ++j) {
            t = fqmul(zeta, r[j + step]);
            r[j + step] = barrett_reduce(r[j] - t);
            r[j]        = barrett_reduce(r[j] + t);
        }
    }
    /* levels 2,1 */
    for (level = 2; level >= 1; level--) {
        step = (1 << level);
        for (start = 0; start < 256; start = j + step) {
            int16_t zeta = ntt_zetas[k++];
            for (j = start; j < start + step; ++j) {
                t = fqmul(zeta, r[j + step]);
                r[j + step] = r[j] - t;
                r[j]        = r[j] + t;
            }
        }
    }
    /* level 0: step=1 */
    step = 1;
    for (start = 0; start < 256; start = j + step) {
        int16_t zeta = ntt_zetas[k++];
        for (j = start; j < start + step; ++j) {
            t = fqmul(zeta, r[j + step]);
            r[j + step] = barrett_reduce(r[j] - t);
            r[j]        = barrett_reduce(r[j] + t);
        }
    }
}

/* Aigis 8 级 INVNTT (串行)
 * 使用 int32_t 中间变量 t 以避免上溢，与 CPU 参考一致 */
static __device__ __noinline__ void invntt(int16_t r[256])
{
    int start, level, step, j, k;
    int32_t t;

    k = 0;
    for (level = 0; level < 7; level++) {
        step = (1 << level);
        for (start = 0; start < 256; start = j + step) {
            int32_t zeta = ntt_zetas_inv[k++];
            for (j = start; j < start + step; ++j) {
                t = r[j];
                if (level & 1)
                    r[j] = barrett_reduce((int16_t)(t + r[j + step]));
                else
                    r[j] = (int16_t)(t + r[j + step]);
                t -= r[j + step];
                r[j + step] = montgomery_reduce((int32_t)zeta * (int16_t)t);
            }
        }
    }
    /* level 7: step=128, 含 N^{-1} 归一化
     * montgomery_reduce(256 * a) = a * 256 * R^{-1} mod Q = a * N^{-1} mod Q */
    step = 128;
    for (start = 0; start < 256; start = j + step) {
        int32_t zeta = ntt_zetas_inv[k++];
        for (j = start; j < start + step; ++j) {
            t = r[j];
            r[j] = montgomery_reduce(256 * (t + r[j + step]));
            t -= r[j + step];
            r[j + step] = montgomery_reduce(zeta * (int16_t)t);
        }
    }
}

/* Aigis 逐点累加 (polyvec 内积)
 * 参考实现: 先将 b 转换到 Montgomery 域 (b*R mod Q)，再做 montgomery_reduce(a * b*R) = a*b mod Q
 * 与参考 pqc_polyvec_pointwise_acc 完全等价 */
static __device__ __noinline__ void polyvec_basemul_acc(kem_poly *r, const kem_polyvec *a, const kem_polyvec *b)
{
    for (int c = 0; c < PARAM_N; c++) {
        /* 先处理 j=0 */
        int16_t t = montgomery_reduce((int32_t)MONT_R2 * b->vec[0].coeffs[c]);
        r->coeffs[c] = montgomery_reduce((int32_t)a->vec[0].coeffs[c] * t);
        /* 累加剩余 j=1..K-1 */
        for (int j = 1; j < PARAM_K; j++) {
            t = montgomery_reduce((int32_t)MONT_R2 * b->vec[j].coeffs[c]);
            r->coeffs[c] += montgomery_reduce((int32_t)a->vec[j].coeffs[c] * t);
        }
        r->coeffs[c] = barrett_reduce(r->coeffs[c]);
    }
}

#endif  /* ALGORITHM */

/* ================================================================
 *  polyvec_ntt / polyvec_invntt — 对向量中每个多项式做 NTT/INVNTT
 * ================================================================ */
static __device__ __noinline__ void polyvec_ntt(kem_polyvec *pv)
{
    for (int i = 0; i < PARAM_K; i++) {
        ntt(pv->vec[i].coeffs);
#if ALGORITHM == ALGO_KYBER
        /* Kyber NTT 不在内部归一化 (级间无 Barrett reduce).
         * 参考实现 poly_ntt() 总是在 ntt() 后调用 poly_reduce().
         * 不做这一步，NTT 输出可达 ±8Q，导致 fqmul 时 Montgomery 越界. */
        for (int j = 0; j < PARAM_N; j++)
            pv->vec[i].coeffs[j] = barrett_reduce(pv->vec[i].coeffs[j]);
#endif
    }
}

static __device__ __noinline__ void polyvec_invntt(kem_polyvec *pv)
{
    for (int i = 0; i < PARAM_K; i++) invntt(pv->vec[i].coeffs);
}

static __device__ __noinline__ void poly_invntt(kem_poly *p)
{
    invntt(p->coeffs);
}

#endif /* NTT_CUH */
