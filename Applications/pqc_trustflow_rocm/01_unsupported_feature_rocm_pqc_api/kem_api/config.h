/*
 * config.h — 算法选择
 *
 * 设置方法: 编译时传入 -DALGORITHM=ALGO_KYBER 或 -DALGORITHM=ALGO_AIGIS_ENC
 *           以及 -DPARAM_MODE=<mode>
 *
 * Kyber 模式:
 *   -DPARAM_MODE=2  -> Kyber512  (K=2)
 *   -DPARAM_MODE=3  -> Kyber768  (K=3)
 *   -DPARAM_MODE=4  -> Kyber1024 (K=4)
 *
 * Aigis-enc 模式:
 *   -DPARAM_MODE=1  -> Aigis-enc-1 (K=2)
 *   -DPARAM_MODE=2  -> Aigis-enc-2 (K=3, low)
 *   -DPARAM_MODE=3  -> Aigis-enc-3 (K=3, med)
 *   -DPARAM_MODE=4  -> Aigis-enc-4 (K=4, high)
 */

#ifndef CONFIG_H
#define CONFIG_H

#define ALGO_KYBER     1
#define ALGO_AIGIS_ENC 2

#ifndef ALGORITHM
#define ALGORITHM  ALGO_KYBER
#endif

#ifndef PARAM_MODE
#if ALGORITHM == ALGO_KYBER
#define PARAM_MODE  3   /* Kyber768 */
#else
#define PARAM_MODE  4   /* Aigis-enc-4 */
#endif
#endif

#if ALGORITHM != ALGO_KYBER && ALGORITHM != ALGO_AIGIS_ENC
#error "ALGORITHM must be ALGO_KYBER or ALGO_AIGIS_ENC"
#endif

#endif /* CONFIG_H */
