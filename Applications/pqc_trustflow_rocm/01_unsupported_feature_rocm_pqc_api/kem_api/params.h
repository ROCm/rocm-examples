/*
 * params.h — 统一参数头文件
 *
 * Kyber (CRYSTALS-Kyber) 和 Aigis-enc (PQMagic KEM) 通过同一套宏描述参数。
 * 两种算法均使用 int16_t 系数，N=256，结构相同。
 *
 * 关键算法差异:
 *   NTT 阶数:  Kyber=7级+basemul (Q≡1 mod 256), Aigis=8级+逐点 (Q≡1 mod 256)
 *   PK 打包:   Kyber=12-bit tobytes, Aigis=压缩至 BITS_PK bits
 *   CT 符号:   Kyber v = pk*r + e2 + msg；Aigis v = pk*r + e2 - msg
 *   拒绝采样:  Kyber=12-bit (4096), Aigis=13-bit (8192)
 *
 * Kyber 参数来源: CRYSTALS-Kyber specification (NIST PQC Round 3)
 * Aigis-enc 参数来源: PQMagic CPU 实现 (AIGIS_ENC_MODE=1/2/3/4)
 */

#ifndef PARAMS_H
#define PARAMS_H

#include "config.h"
#include <stdint.h>

/* ================================================================
 *  通用类型和常量
 * ================================================================ */
typedef int16_t coeff_t;   /* 两种算法均适用 (Q < 2^13 < 2^15) */

#define PARAM_N      256
#define PARAM_SYMBYTES 32
#define PARAM_SSBYTES  32

/* ================================================================
 *  Kyber 参数
 * ================================================================ */
#if ALGORITHM == ALGO_KYBER

#define PARAM_Q       3329
#define PARAM_QBITS   12     /* 拒绝采样位宽 */
#define PARAM_QINV    62209  /* Q^{-1} mod 2^16 (used as int16 signed = -3327) */

/* Montgomery 常数: R=2^16
 * MONT_R2 = R^2 mod Q = 1353 (用于转换到 Mont 域) */
#define MONT_R2       1353

#define PARAM_ETA2    2      /* 加密误差 eta */

#if PARAM_MODE == 2   /* Kyber512 */
  #define PARAM_K          2
  #define PARAM_ETA1       3
  #define PARAM_BITS_PK    12   /* pk = polyvec_tobytes12 (无压缩损失) */
  #define PARAM_BITS_C1    10   /* ct 向量压缩位数 */
  #define PARAM_BITS_C2    4    /* ct 标量多项式压缩位数 */
  #define CRYPTO_ALGNAME   "Kyber512"

#elif PARAM_MODE == 3 /* Kyber768 */
  #define PARAM_K          3
  #define PARAM_ETA1       2
  #define PARAM_BITS_PK    12
  #define PARAM_BITS_C1    10
  #define PARAM_BITS_C2    4
  #define CRYPTO_ALGNAME   "Kyber768"

#elif PARAM_MODE == 4 /* Kyber1024 */
  #define PARAM_K          4
  #define PARAM_ETA1       2
  #define PARAM_BITS_PK    12
  #define PARAM_BITS_C1    11
  #define PARAM_BITS_C2    5
  #define CRYPTO_ALGNAME   "Kyber1024"

#else
  #error "PARAM_MODE must be 2, 3, or 4 for Kyber"
#endif

/* Kyber 噪声 eta: s 和 e 用 ETA1，加密噪声用 ETA1/ETA2 */
#define PARAM_ETA_S     PARAM_ETA1
#define PARAM_ETA_E_KG  PARAM_ETA1   /* 密钥生成误差 */
#define PARAM_ETA_E_ENC PARAM_ETA1   /* 加密误差 (e1) */
#define PARAM_ETA_E2    PARAM_ETA2   /* 加密标量误差 (e2) */

/* Kyber 全精度多项式字节数 (12-bit * 256 = 384 bytes) */
#define PARAM_POLYBYTES         384

/* Kyber PRF 输出长度 (bytes): ETA * N / 4 */
#define PARAM_PRF_ETA1_BYTES    (PARAM_ETA1 * PARAM_N / 4)
#define PARAM_PRF_ETA2_BYTES    (PARAM_ETA2 * PARAM_N / 4)

/* ================================================================
 *  Aigis-enc 参数
 * ================================================================ */
#elif ALGORITHM == ALGO_AIGIS_ENC

#define PARAM_Q       7681
#define PARAM_QBITS   13     /* 拒绝采样位宽 */
#define PARAM_QINV    57857  /* Q^{-1} mod 2^16 */

#define MONT_R2       5569   /* R^2 mod Q */

#if PARAM_MODE == 1   /* Aigis-enc-1 (K=2) */
  #define PARAM_K          2
  #define PARAM_ETA_S      4
  #define PARAM_ETA_E_KG   8
  #define PARAM_ETA_E_ENC  8
  #define PARAM_ETA_E2     8
  #define PARAM_BITS_PK    10
  #define PARAM_BITS_C1    10
  #define PARAM_BITS_C2    3
  #define CRYPTO_ALGNAME   "Aigis-enc-1"

#elif PARAM_MODE == 2 /* Aigis-enc-2 (K=3, low) */
  #define PARAM_K          3
  #define PARAM_ETA_S      1
  #define PARAM_ETA_E_KG   4
  #define PARAM_ETA_E_ENC  4
  #define PARAM_ETA_E2     4
  #define PARAM_BITS_PK    9
  #define PARAM_BITS_C1    9
  #define PARAM_BITS_C2    4
  #define CRYPTO_ALGNAME   "Aigis-enc-2"

#elif PARAM_MODE == 3 /* Aigis-enc-3 (K=3, med) */
  #define PARAM_K          3
  #define PARAM_ETA_S      2
  #define PARAM_ETA_E_KG   4
  #define PARAM_ETA_E_ENC  4
  #define PARAM_ETA_E2     4
  #define PARAM_BITS_PK    10
  #define PARAM_BITS_C1    10
  #define PARAM_BITS_C2    3
  #define CRYPTO_ALGNAME   "Aigis-enc-3"

#elif PARAM_MODE == 4 /* Aigis-enc-4 (K=4, high) */
  #define PARAM_K          4
  #define PARAM_ETA_S      3
  #define PARAM_ETA_E_KG   8
  #define PARAM_ETA_E_ENC  8
  #define PARAM_ETA_E2     8
  #define PARAM_BITS_PK    11
  #define PARAM_BITS_C1    11
  #define PARAM_BITS_C2    5
  #define CRYPTO_ALGNAME   "Aigis-enc-4"

#else
  #error "PARAM_MODE must be 1, 2, 3, or 4 for Aigis-enc"
#endif

/* Aigis 全精度多项式字节数 (13-bit * 256 = 416 bytes) */
#define PARAM_POLYBYTES         416

/* Aigis PRF 输出长度 (用最大 eta 覆盖; 实际只需 eta*N/4 字节) */
#define PARAM_PRF_ETA1_BYTES    (PARAM_ETA_S * 64)
#define PARAM_PRF_ETA2_BYTES    (PARAM_ETA_E_KG * 64)

#endif /* ALGORITHM */

/* ================================================================
 *  派生常量 (两种算法共用公式)
 * ================================================================ */
#define PARAM_POLYVECBYTES       (PARAM_K * PARAM_POLYBYTES)
#define PARAM_PK_POLYVEC_BYTES   (PARAM_BITS_PK * PARAM_K * PARAM_N / 8)
#define PARAM_CT_VEC_BYTES       (PARAM_BITS_C1 * PARAM_K * PARAM_N / 8)
#define PARAM_CT_POLY_BYTES      (PARAM_BITS_C2 * PARAM_N / 8)

#define PARAM_PUBLICKEYBYTES     (PARAM_PK_POLYVEC_BYTES + PARAM_SYMBYTES)
#define PARAM_INDCPA_SECRETKEYBYTES  PARAM_POLYVECBYTES
#define PARAM_SECRETKEYBYTES     (PARAM_POLYVECBYTES + PARAM_PUBLICKEYBYTES + 2 * PARAM_SYMBYTES)
#define PARAM_CIPHERTEXTBYTES    (PARAM_CT_VEC_BYTES + PARAM_CT_POLY_BYTES)

/* 矩阵生成 XOF 缓冲区大小 */
#define PARAM_GEN_MATRIX_NBLOCKS  4
#define PARAM_XOF_BLOCKBYTES      168  /* SHAKE128_RATE */
#define PARAM_GEN_MATRIX_BUFLEN   (PARAM_GEN_MATRIX_NBLOCKS * PARAM_XOF_BLOCKBYTES)

/* 最大 K (用于固定大小的 struct) */
#define MAX_K  4

/* ================================================================
 *  多项式结构体
 * ================================================================ */
typedef struct { int16_t coeffs[PARAM_N]; } kem_poly;
typedef struct { kem_poly vec[MAX_K]; }     kem_polyvec;

#endif /* PARAMS_H */
