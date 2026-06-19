#include "hip/hip_runtime.h"
/*
 * params.h — 统一参数头文件
 *
 * 两种算法通过同一套宏名描述所有参数。
 * 算法语义差异 (系数域, 采样, challenge, hint 等) 通过
 * #if ALGORITHM 分支在各功能文件中处理。
 *
 * 关键统一决策:
 *   coeff_t = int32_t  (ML-DSA 本已是signed; Aigis系数 < Q < 4M < 2^22, 完全适配)
 *   PARAM_ETA_S1/S2    = s1/s2 多项式的 eta (Aigis中 ETA1 ≠ ETA2)
 *   PARAM_BETA1/BETA2  = TAU * ETA_S1/S2  (norm reject 阈值)
 *   TRBYTES = CRHBYTES (签名的 tr 长度等于哈希输出长度)
 *   RNDBYTES           = 随机化签名熵 (ML-DSA=32, Aigis=0)
 *   SETA1BITS/SETA2BITS = bits to pack PARAM_ETA_S1/S2 coefficients
 *   POLYT1_PACKED_BITS  = QBITS - D (bits per t1 coeff)
 *
 * Aigis参数来源: PQMagic CPU实现 (PARAMS=1/2/3)
 *   Mode 1: Q=2021377, K=4, L=3, ETA1=2, ETA2=3, D=13, GAMMA1=2^17, GAMMA2=(Q-1)/12
 *   Mode 2: Q=3870721, K=5, L=4, ETA1=2, ETA2=5, D=14, GAMMA1=2^17, GAMMA2=(Q-1)/12
 *   Mode 3: Q=3870721, K=6, L=5, ETA1=1, ETA2=5, D=14, GAMMA1=2^17, GAMMA2=(Q-1)/12
 *
 * 签名格式:
 *   ML-DSA: c_tilde (CTILDEBYTES) || z_packed || hint_bitmap
 *   Aigis:  z_packed || hint_bitmap || challenge_poly (N/8 + 8 bytes)
 */

#ifndef PARAMS_H
#define PARAMS_H

#include "config.h"
#include <stdint.h>

/* ================================================================
 *  通用类型 — 对两种算法均为 int32_t
 * ================================================================ */
typedef int32_t coeff_t;
typedef int64_t coeff2_t;

/* ================================================================
 *  通用常量
 * ================================================================ */
#define PARAM_N      256
#define SEEDBYTES    32

/* ================================================================
 *  ML-DSA (CRYSTALS-Dilithium) 参数
 * ================================================================ */
#if ALGORITHM == ALGO_MLDSA

#define PARAM_Q      8380417
#define PARAM_QBITS  23
#define CRHBYTES     64
#define TRBYTES      64
#define RNDBYTES     32

/* Mont constants for Q=8380417:
 *   MONT_VAL = 2^32 mod Q = 4193792
 *   MONT_QINV: Q^{-1} mod 2^32 = 58728449 (fits in uint32) */
#define MONT_VAL     4193792
#define MONT_QINV    58728449u

#if PARAM_MODE == 2   /* ML-DSA-44 */
  #define PARAM_K        4
  #define PARAM_L        4
  #define PARAM_D        13
  #define PARAM_ETA_S1   2
  #define PARAM_ETA_S2   2
  #define PARAM_TAU      39
  #define PARAM_BETA1    78           /* TAU * ETA_S1 */
  #define PARAM_BETA2    78           /* TAU * ETA_S2 */
  #define PARAM_GAMMA1   (1 << 17)
  #define PARAM_GAMMA2   ((PARAM_Q - 1) / 88)
  #define PARAM_OMEGA    80
  #define CTILDEBYTES    32
  #define SETA1BITS      3     /* ceil(log2(2*2+1)) = ceil(log2(5)) = 3 */
  #define SETA2BITS      3
  #define INTT_F         41978 /* N^{-1} * 2^32 mod Q */

#elif PARAM_MODE == 3 /* ML-DSA-65 */
  #define PARAM_K        6
  #define PARAM_L        5
  #define PARAM_D        13
  #define PARAM_ETA_S1   4
  #define PARAM_ETA_S2   4
  #define PARAM_TAU      49
  #define PARAM_BETA1    196
  #define PARAM_BETA2    196
  #define PARAM_GAMMA1   (1 << 19)
  #define PARAM_GAMMA2   ((PARAM_Q - 1) / 32)
  #define PARAM_OMEGA    55
  #define CTILDEBYTES    48
  #define SETA1BITS      4     /* ceil(log2(2*4+1)) = ceil(log2(9)) = 4 */
  #define SETA2BITS      4
  #define INTT_F         41978

#elif PARAM_MODE == 5 /* ML-DSA-87 */
  #define PARAM_K        8
  #define PARAM_L        7
  #define PARAM_D        13
  #define PARAM_ETA_S1   2
  #define PARAM_ETA_S2   2
  #define PARAM_TAU      60
  #define PARAM_BETA1    120
  #define PARAM_BETA2    120
  #define PARAM_GAMMA1   (1 << 19)
  #define PARAM_GAMMA2   ((PARAM_Q - 1) / 32)
  #define PARAM_OMEGA    75
  #define CTILDEBYTES    64
  #define SETA1BITS      3
  #define SETA2BITS      3
  #define INTT_F         41978
#endif

#define CRYPTO_ALGNAME  "ML-DSA"

/* ================================================================
 *  Aigis-sig (PQMagic) 参数
 *  来源: PQMagic GPU实现 params.h (PARAMS=1/2/3)
 * ================================================================ */
#elif ALGORITHM == ALGO_AIGIS

#define CRHBYTES     48
#define TRBYTES      48
#define RNDBYTES     0

/* Aigis ALPHA = 2*GAMMA2 — used in decompose/use_hint */
#define PARAM_ALPHA_VAL   (2 * ((PARAM_Q - 1) / 12))

#if PARAM_MODE == 1   /* Aigis-sig1 */
  #define PARAM_Q       2021377
  #define PARAM_QBITS   21
  #define PARAM_K       4
  #define PARAM_L       3
  #define PARAM_D       13
  #define PARAM_ETA_S1  2
  #define PARAM_ETA_S2  3
  #define PARAM_TAU     60
  #define PARAM_BETA1   120         /* TAU * ETA_S1 = 60*2 */
  #define PARAM_BETA2   175         /* from PQMagic params: 175 (~TAU*ETA_S2 slightly adjusted) */
  #define PARAM_GAMMA1  (1 << 17)
  #define PARAM_GAMMA2  ((PARAM_Q - 1) / 12)   /* = 168448 */
  #define PARAM_OMEGA   80
  #define SETA1BITS     3     /* ceil(log2(2*2+1))=3 */
  #define SETA2BITS     3     /* ceil(log2(2*3+1))=3 */
  /* Mont: 2^32 mod Q=2021377 = 1562548; Q^{-1} mod 2^32 */
  #define MONT_VAL      1562548
  #define MONT_QINV     1445013505u

#elif PARAM_MODE == 2 /* Aigis-sig2 */
  #define PARAM_Q       3870721
  #define PARAM_QBITS   22
  #define PARAM_K       5
  #define PARAM_L       4
  #define PARAM_D       14
  #define PARAM_ETA_S1  2
  #define PARAM_ETA_S2  5
  #define PARAM_TAU     60
  #define PARAM_BETA1   120         /* TAU * ETA_S1 = 60*2 */
  #define PARAM_BETA2   275         /* from PQMagic params */
  #define PARAM_GAMMA1  (1 << 17)
  #define PARAM_GAMMA2  ((PARAM_Q - 1) / 12)   /* = 322560 */
  #define PARAM_OMEGA   96
  #define SETA1BITS     3     /* ceil(log2(5))=3 */
  #define SETA2BITS     4     /* ceil(log2(11))=4 */
  /* Mont: 2^32 mod Q=3870721 = 2337707; Q^{-1} mod 2^32 */
  #define MONT_VAL      2337707
  #define MONT_QINV     1623519233u

#elif PARAM_MODE == 3 /* Aigis-sig3 */
  #define PARAM_Q       3870721
  #define PARAM_QBITS   22
  #define PARAM_K       6
  #define PARAM_L       5
  #define PARAM_D       14
  #define PARAM_ETA_S1  1
  #define PARAM_ETA_S2  5
  #define PARAM_TAU     60
  #define PARAM_BETA1   60          /* TAU * ETA_S1 = 60*1 */
  #define PARAM_BETA2   275         /* from PQMagic params */
  #define PARAM_GAMMA1  (1 << 17)
  #define PARAM_GAMMA2  ((PARAM_Q - 1) / 12)   /* = 322560 */
  #define PARAM_OMEGA   120
  #define SETA1BITS     2     /* ceil(log2(3))=2: values {-1,0,1}→{2,1,0} */
  #define SETA2BITS     4     /* ceil(log2(11))=4 */
  /* Mont: same as mode 2; Q^{-1} mod 2^32 */
  #define MONT_VAL      2337707
  #define MONT_QINV     1623519233u
#endif

#define CRYPTO_ALGNAME  "Aigis-sig"

#endif /* ALGORITHM */

/* ================================================================
 *  算法钩子宏 — 消除 poly/polyvec 层的大量 #if ALGORITHM 分叉
 * ================================================================ */

/*
 * COEFF_BIAS: 系数偏置常量
 *   ML-DSA 使用中心化 (-Q/2, Q/2], 偏置 = 0
 *   Aigis  使用无符号 [0, Q),      偏置 = Q
 *   eta/t0 pack/unpack 统一为: COEFF_BIAS + ETA - coeff
 */
#if ALGORITHM == ALGO_MLDSA
#define COEFF_BIAS   0
#elif ALGORITHM == ALGO_AIGIS
#define COEFF_BIAS   PARAM_Q
#endif

/*
 * MATRIX_NONCE(i,j): matrix A expansion 的 nonce 编码
 *   ML-DSA: 2-byte LE,  nonce = i*256 + j
 *   Aigis:  1-byte,      nonce = i + (j<<4)
 */
#if ALGORITHM == ALGO_MLDSA
#define MATRIX_NONCE(i, j) ((uint16_t)((i) * 256 + (j)))
#elif ALGORITHM == ALGO_AIGIS
#define MATRIX_NONCE(i, j) ((uint16_t)((i) + ((j) << 4)))
#endif

/*
 * GAMMA1_NONCE(base, i): gamma1 采样的 nonce 计算
 *   ML-DSA: nonce = L * base + i (每次 rejection 只递增 base 一次)
 *   Aigis:  nonce = base + i     (每个 poly 一个 nonce)
 */
#if ALGORITHM == ALGO_MLDSA
#define GAMMA1_NONCE(base, i)  ((uint16_t)(PARAM_L * (base) + (i)))
#elif ALGORITHM == ALGO_AIGIS
#define GAMMA1_NONCE(base, i)  ((uint16_t)((base) + (i)))
#endif

/*
 * Z_BIAS / Z_FIXUP(t): polyz pack/unpack 的偏置
 *   ML-DSA: t = GAMMA1 - coeff,             Z_FIXUP 为空
 *   Aigis:  t = GAMMA1-1 - coeff; 负值 +Q,  Z_FIXUP 修正负值
 */
#if ALGORITHM == ALGO_MLDSA
#define Z_BIAS        PARAM_GAMMA1
#define Z_FIXUP(t)    /* nothing */
#elif ALGORITHM == ALGO_AIGIS
#define Z_BIAS        (PARAM_GAMMA1 - 1)
#define Z_FIXUP(t)    (t) += (((int32_t)(t)) >> 31) & PARAM_Q
#endif

/* ================================================================
 *  导出的打包尺寸 (基于参数计算, 两种算法通用公式)
 * ================================================================ */

/* bits per t1 coeff: POLYT1_PACKED_BITS = QBITS - D */
#define POLYT1_PACKED_BITS   (PARAM_QBITS - PARAM_D)
/* bytes per poly t1: N * bits / 8 */
#define POLYT1_PACKEDBYTES   (PARAM_N * POLYT1_PACKED_BITS / 8)

/* bytes per poly t0: N * D / 8 */
#define POLYT0_PACKEDBYTES   (PARAM_N * PARAM_D / 8)

/* bytes per eta poly (s1): N * SETA1BITS / 8 */
#define POLYETA1_PACKEDBYTES (PARAM_N * SETA1BITS / 8)

/* bytes per eta poly (s2): N * SETA2BITS / 8 */
#define POLYETA2_PACKEDBYTES (PARAM_N * SETA2BITS / 8)

/* bytes per z poly: depends on GAMMA1 (18-bit or 20-bit coeffs) */
#if PARAM_GAMMA1 == (1 << 17)
#define POLYZ_PACKEDBYTES    576   /* 9 bytes per 4 coeffs (18 bits) */
#elif PARAM_GAMMA1 == (1 << 19)
#define POLYZ_PACKEDBYTES    640   /* 5 bytes per 2 coeffs (20 bits) */
#endif

/* bytes per w1 poly: depends on bits per coeff = ceil(log2(N_W1+1))
 *   GAMMA2=(Q-1)/88 → N_W1=44 → 6 bits/coeff → 4 per 3 bytes → 192
 *   GAMMA2=(Q-1)/32 → N_W1=16 → 4 bits/coeff → 2 per 1 byte  → 128
 *   GAMMA2=(Q-1)/12 → N_W1=6  → 3 bits/coeff → 8 per 3 bytes → 96
 */
#if PARAM_GAMMA2 == (PARAM_Q - 1) / 88
#define POLYW1_PACKEDBYTES   192
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 32
#define POLYW1_PACKEDBYTES   128
#elif PARAM_GAMMA2 == (PARAM_Q - 1) / 12
#define POLYW1_PACKEDBYTES   96
#endif

/* Number of distinct high-bit parts: N_W1 = (Q-1) / (2 * GAMMA2) */
#define N_W1   ((PARAM_Q - 1) / (2 * PARAM_GAMMA2))

/* Public/Secret key and Signature sizes */
#define CRYPTO_PUBLICKEYBYTES  (SEEDBYTES + PARAM_K * POLYT1_PACKEDBYTES)
#define CRYPTO_SECRETKEYBYTES  (2 * SEEDBYTES + TRBYTES \
    + PARAM_L * POLYETA1_PACKEDBYTES \
    + PARAM_K * POLYETA2_PACKEDBYTES \
    + PARAM_K * POLYT0_PACKEDBYTES)

#if ALGORITHM == ALGO_MLDSA
/* ML-DSA sig format: c_tilde || z_packed || hints_bitmap */
#define CRYPTO_BYTES           (CTILDEBYTES \
    + PARAM_L * POLYZ_PACKEDBYTES \
    + PARAM_OMEGA + PARAM_K)
#elif ALGORITHM == ALGO_AIGIS
/* Aigis sig format: z_packed || hints_bitmap || challenge_poly (N/8+8 bytes) */
#define CHALLENGE_POLY_PACKEDBYTES  (PARAM_N / 8 + 8)   /* 40 bytes: bitmap + signs */
#define CRYPTO_BYTES           (PARAM_L * POLYZ_PACKEDBYTES \
    + PARAM_OMEGA + PARAM_K \
    + CHALLENGE_POLY_PACKEDBYTES)
#endif

#endif /* PARAMS_H */
