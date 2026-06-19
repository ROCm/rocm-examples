/*
 * config.h — 算法选择
 *
 * 设置方法: 编译时传入 -DALGORITHM=ALGO_MLDSA 或 -DALGORITHM=ALGO_AIGIS
 *           以及 -DPARAM_MODE=2/3/5 (ML-DSA) 或 -DPARAM_MODE=1/2/3 (Aigis)
 */

#ifndef CONFIG_H
#define CONFIG_H

#define ALGO_MLDSA  1
#define ALGO_AIGIS  2

#ifndef ALGORITHM
#define ALGORITHM   ALGO_MLDSA
#endif

#ifndef PARAM_MODE
#if ALGORITHM == ALGO_MLDSA
#define PARAM_MODE  5   /* ML-DSA-87 */
#else
#define PARAM_MODE  3   /* Aigis-sig3 */
#endif
#endif

/* 编译期检查 */
#if ALGORITHM != ALGO_MLDSA && ALGORITHM != ALGO_AIGIS
#error "ALGORITHM must be ALGO_MLDSA or ALGO_AIGIS"
#endif

#endif /* CONFIG_H */
