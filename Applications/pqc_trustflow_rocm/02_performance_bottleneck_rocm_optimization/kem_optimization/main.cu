/*
 * main.cu — 统一 KEM 测试驱动程序 (Kyber + Aigis-enc)
 *
 * 编译示例:
 *   nvcc -O2 -DALGORITHM=1 -DPARAM_MODE=3 -o kyber768.exe   main.cu
 *   nvcc -O2 -DALGORITHM=2 -DPARAM_MODE=3 -o aigisenc3.exe  main.cu
 *
 * 用法:
 *   kyber768.exe                  — 运行正确性测试 + 默认批量吞吐量测试
 *   kyber768.exe --batch 8192     — 指定批量大小
 *   kyber768.exe --sweep          — 扫描不同 batch size
 *   kyber768.exe --serial-only    — 仅运行串行设备函数 (不用流水线 kernel)
 */

#include "rocm_compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "params.h"
#include "batch_kem.cuh"

/* ================================================================
 *  工具宏
 * ================================================================ */
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

static double get_time_ms(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ================================================================
 *  算法名称
 * ================================================================ */
static const char *algo_name(void)
{
#if ALGORITHM == ALGO_KYBER
    #if   PARAM_MODE == 2
        return "Kyber-512";
    #elif PARAM_MODE == 3
        return "Kyber-768";
    #else
        return "Kyber-1024";
    #endif
#elif ALGORITHM == ALGO_AIGIS_ENC
    #if   PARAM_MODE == 1
        return "Aigis-enc-1";
    #elif PARAM_MODE == 2
        return "Aigis-enc-2";
    #elif PARAM_MODE == 3
        return "Aigis-enc-3";
    #else
        return "Aigis-enc-4";
    #endif
#endif
}

/* ================================================================
 *  正确性测试: 单实例 CPU 调用 GPU kernel 验证
 * ================================================================ */
static int test_correctness(void)
{
    printf("=== 正确性测试: %s ===\n", algo_name());
    printf("  PK=%u SK=%u CT=%u SS=%u 字节\n",
           PARAM_PUBLICKEYBYTES, PARAM_SECRETKEYBYTES,
           PARAM_CIPHERTEXTBYTES, PARAM_SSBYTES);

    /* Host 端分配 */
    uint8_t *h_pk  = (uint8_t *)malloc(PARAM_PUBLICKEYBYTES);
    uint8_t *h_sk  = (uint8_t *)malloc(PARAM_SECRETKEYBYTES);
    uint8_t *h_ct  = (uint8_t *)malloc(PARAM_CIPHERTEXTBYTES);
    uint8_t *h_ss1 = (uint8_t *)malloc(PARAM_SSBYTES);
    uint8_t *h_ss2 = (uint8_t *)malloc(PARAM_SSBYTES);
    uint8_t *h_coins_kg  = (uint8_t *)malloc(2 * PARAM_SYMBYTES);
    uint8_t *h_coins_enc = (uint8_t *)malloc(PARAM_SYMBYTES);

    if (!h_pk || !h_sk || !h_ct || !h_ss1 || !h_ss2 || !h_coins_kg || !h_coins_enc) {
        fprintf(stderr, "malloc failed\n");
        return -1;
    }

    /* 生成伪随机种子 (测试用，实际应用请使用安全随机源) */
    srand(42);
    for (int i = 0; i < 2 * PARAM_SYMBYTES; i++)
        h_coins_kg[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < PARAM_SYMBYTES; i++)
        h_coins_enc[i] = (uint8_t)(rand() & 0xFF);

    /* Device 端分配 */
    uint8_t *d_pk, *d_sk, *d_ct, *d_ss1, *d_ss2;
    uint8_t *d_coins_kg, *d_coins_enc;
    CUDA_CHECK(cudaMalloc(&d_pk,  PARAM_PUBLICKEYBYTES));
    CUDA_CHECK(cudaMalloc(&d_sk,  PARAM_SECRETKEYBYTES));
    CUDA_CHECK(cudaMalloc(&d_ct,  PARAM_CIPHERTEXTBYTES));
    CUDA_CHECK(cudaMalloc(&d_ss1, PARAM_SSBYTES));
    CUDA_CHECK(cudaMalloc(&d_ss2, PARAM_SSBYTES));
    CUDA_CHECK(cudaMalloc(&d_coins_kg,  2 * PARAM_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&d_coins_enc, PARAM_SYMBYTES));

    CUDA_CHECK(cudaMemcpy(d_coins_kg,  h_coins_kg,  2 * PARAM_SYMBYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coins_enc, h_coins_enc, PARAM_SYMBYTES,     cudaMemcpyHostToDevice));

    /* 串行设备 kernel 验证 (batch_count=1) */
    batch_kem_keypair_serial_kernel<<<1, 1>>>(d_pk, d_sk, d_coins_kg, 1);
    CUDA_CHECK(cudaGetLastError());
    batch_kem_encaps_serial_kernel<<<1, 1>>>(d_ct, d_ss1, d_pk, d_coins_enc, 1);
    CUDA_CHECK(cudaGetLastError());
    batch_kem_decaps_serial_kernel<<<1, 1>>>(d_ss2, d_ct, d_sk, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 取回结果 */
    CUDA_CHECK(cudaMemcpy(h_ss1, d_ss1, PARAM_SSBYTES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ss2, d_ss2, PARAM_SSBYTES, cudaMemcpyDeviceToHost));

    /* 验证 ss1 == ss2 */
    int ok = (memcmp(h_ss1, h_ss2, PARAM_SSBYTES) == 0);
    printf("  KEM 正确性: %s\n", ok ? "PASS" : "FAIL");

    if (!ok) {
        printf("  [encaps ss]  ");
        for (int i = 0; i < 8; i++) printf("%02x", h_ss1[i]);
        printf("...\n");
        printf("  [decaps ss]  ");
        for (int i = 0; i < 8; i++) printf("%02x", h_ss2[i]);
        printf("...\n");
    }

    cudaFree(d_pk); cudaFree(d_sk); cudaFree(d_ct);
    cudaFree(d_ss1); cudaFree(d_ss2);
    cudaFree(d_coins_kg); cudaFree(d_coins_enc);
    free(h_pk); free(h_sk); free(h_ct);
    free(h_ss1); free(h_ss2);
    free(h_coins_kg); free(h_coins_enc);

    return ok ? 0 : 1;
}

/* ================================================================
 *  批量吞吐量测试
 * ================================================================ */
static void bench_batch(int batch_count, int n_ops, int use_pipeline, int profile_pipeline = 0)
{
    printf("\n--- batch=%d n_ops=%d mode=%s ---\n",
           batch_count, n_ops, use_pipeline ? "pipeline" : "serial");

    /* 分配设备内存 */
    uint8_t *d_pk, *d_sk, *d_ct, *d_ss;
    uint8_t *d_coins_kg, *d_coins_enc;

    CUDA_CHECK(cudaMalloc(&d_pk,  (size_t)batch_count * PARAM_PUBLICKEYBYTES));
    CUDA_CHECK(cudaMalloc(&d_sk,  (size_t)batch_count * PARAM_SECRETKEYBYTES));
    CUDA_CHECK(cudaMalloc(&d_ct,  (size_t)batch_count * PARAM_CIPHERTEXTBYTES));
    CUDA_CHECK(cudaMalloc(&d_ss,  (size_t)batch_count * PARAM_SSBYTES));
    CUDA_CHECK(cudaMalloc(&d_coins_kg,  (size_t)batch_count * 2 * PARAM_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&d_coins_enc, (size_t)batch_count * PARAM_SYMBYTES));

    /* 生成随机种子 */
    uint8_t *h_coins_kg  = (uint8_t *)malloc((size_t)batch_count * 2 * PARAM_SYMBYTES);
    uint8_t *h_coins_enc = (uint8_t *)malloc((size_t)batch_count * PARAM_SYMBYTES);
    if (!h_coins_kg || !h_coins_enc) { fprintf(stderr, "OOM\n"); exit(1); }

    srand(1234);
    for (size_t i = 0; i < (size_t)batch_count * 2 * PARAM_SYMBYTES; i++)
        h_coins_kg[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < (size_t)batch_count * PARAM_SYMBYTES; i++)
        h_coins_enc[i] = (uint8_t)(rand() & 0xFF);

    CUDA_CHECK(cudaMemcpy(d_coins_kg,  h_coins_kg,  (size_t)batch_count * 2 * PARAM_SYMBYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coins_enc, h_coins_enc, (size_t)batch_count * PARAM_SYMBYTES,     cudaMemcpyHostToDevice));

    BatchKemBuffers buf = {};
    if (use_pipeline) {
        /* 修复 batch_kem_alloc 中的双 cudaMalloc bug: 直接内联分配 */
        buf.max_batch = batch_count;
        CUDA_CHECK(cudaMalloc(&buf.d_mat,  (size_t)PARAM_K * PARAM_K * batch_count * PARAM_N * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_skpv, (size_t)PARAM_K * batch_count * PARAM_N * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_pkpv, (size_t)PARAM_K * batch_count * PARAM_N * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_e,    (size_t)PARAM_K * batch_count * PARAM_N * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_publicseed_kg, (size_t)batch_count * PARAM_SYMBYTES));
        CUDA_CHECK(cudaMalloc(&buf.d_noiseseed_kg, (size_t)batch_count * PARAM_SYMBYTES));
        buf.d_pk_bytes  = d_pk;
        buf.d_sk_bytes  = d_sk;
        buf.d_ct_bytes  = d_ct;
        buf.d_ss_bytes  = d_ss;
        buf.d_coins_kg  = d_coins_kg;
        buf.d_coins_enc = d_coins_enc;
    }

    /* ---- Keygen ---- */
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = get_time_ms();

    for (int op = 0; op < n_ops; op++) {
        if (use_pipeline) {
            if (profile_pipeline && op == 0)
                batch_keygen_pipelined_profile(d_pk, d_sk, &buf, batch_count);
            else
                batch_keygen_pipelined(d_pk, d_sk, &buf, batch_count);
        } else {
            int tpb = KEM_KEYGEN_TPB;
            int blocks = (batch_count + tpb - 1) / tpb;
            batch_kem_keypair_serial_kernel<<<blocks, tpb>>>(d_pk, d_sk, d_coins_kg, batch_count);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double t_kg = (get_time_ms() - t0) / n_ops;
    double ops_kg = batch_count * 1000.0 / t_kg;
    printf("  Keygen:  %7.1f ms/batch → %.0f ops/sec\n", t_kg, ops_kg);

    /* ---- Encaps ---- */
    t0 = get_time_ms();
    for (int op = 0; op < n_ops; op++) {
        if (use_pipeline) {
            batch_encaps_serial(d_ct, d_ss, d_pk, &buf, batch_count);
        } else {
            int tpb = KEM_ENCAPS_TPB;
            int blocks = (batch_count + tpb - 1) / tpb;
            batch_kem_encaps_serial_kernel<<<blocks, tpb>>>(d_ct, d_ss, d_pk, d_coins_enc, batch_count);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double t_enc = (get_time_ms() - t0) / n_ops;
    double ops_enc = batch_count * 1000.0 / t_enc;
    printf("  Encaps:  %7.1f ms/batch → %.0f ops/sec\n", t_enc, ops_enc);

    /* ---- Decaps ---- */
    t0 = get_time_ms();
    for (int op = 0; op < n_ops; op++) {
        batch_decaps_serial(d_ss, d_ct, d_sk, batch_count);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double t_dec = (get_time_ms() - t0) / n_ops;
    double ops_dec = batch_count * 1000.0 / t_dec;
    printf("  Decaps:  %7.1f ms/batch → %.0f ops/sec\n", t_dec, ops_dec);

    /* 清理 */
    if (use_pipeline) {
        cudaFree(buf.d_mat);
        cudaFree(buf.d_skpv);
        cudaFree(buf.d_pkpv);
        cudaFree(buf.d_e);
        cudaFree(buf.d_publicseed_kg);
        cudaFree(buf.d_noiseseed_kg);
    }
    cudaFree(d_pk); cudaFree(d_sk); cudaFree(d_ct); cudaFree(d_ss);
    cudaFree(d_coins_kg); cudaFree(d_coins_enc);
    free(h_coins_kg); free(h_coins_enc);
}

static void run_serial_kem_round(
    uint8_t *d_pk, uint8_t *d_sk, uint8_t *d_ct, uint8_t *d_ss,
    uint8_t *d_coins_kg, uint8_t *d_coins_enc,
    int batch_count, int n_ops)
{
    int kg_tpb = KEM_KEYGEN_TPB;
    int kg_blocks = (batch_count + kg_tpb - 1) / kg_tpb;
    int enc_tpb = KEM_ENCAPS_TPB;
    int enc_blocks = (batch_count + enc_tpb - 1) / enc_tpb;
    int dec_tpb = KEM_DECAPS_TPB;
    int dec_blocks = (batch_count + dec_tpb - 1) / dec_tpb;

    for (int op = 0; op < n_ops; op++) {
        batch_kem_keypair_serial_kernel<<<kg_blocks, kg_tpb>>>(d_pk, d_sk, d_coins_kg, batch_count);
        batch_kem_encaps_serial_kernel<<<enc_blocks, enc_tpb>>>(d_ct, d_ss, d_pk, d_coins_enc, batch_count);
        batch_kem_decaps_serial_kernel<<<dec_blocks, dec_tpb>>>(d_ss, d_ct, d_sk, batch_count);
    }
}

static void bench_reuse_buffers(int batch_count, int rounds, int n_ops)
{
    printf("\n=== Buffer reuse benchmark: %s ===\n", algo_name());
    printf("batch=%d rounds=%d n_ops_per_round=%d\n", batch_count, rounds, n_ops);

    size_t pk_bytes = (size_t)batch_count * PARAM_PUBLICKEYBYTES;
    size_t sk_bytes = (size_t)batch_count * PARAM_SECRETKEYBYTES;
    size_t ct_bytes = (size_t)batch_count * PARAM_CIPHERTEXTBYTES;
    size_t ss_bytes = (size_t)batch_count * PARAM_SSBYTES;
    size_t kg_bytes = (size_t)batch_count * 2 * PARAM_SYMBYTES;
    size_t enc_bytes = (size_t)batch_count * PARAM_SYMBYTES;

    uint8_t *h_coins_kg = (uint8_t *)malloc(kg_bytes);
    uint8_t *h_coins_enc = (uint8_t *)malloc(enc_bytes);
    if (!h_coins_kg || !h_coins_enc) { fprintf(stderr, "OOM\n"); exit(1); }
    srand(9102);
    for (size_t i = 0; i < kg_bytes; i++) h_coins_kg[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < enc_bytes; i++) h_coins_enc[i] = (uint8_t)(rand() & 0xFF);

    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = get_time_ms();
    for (int r = 0; r < rounds; r++) {
        uint8_t *d_pk, *d_sk, *d_ct, *d_ss, *d_coins_kg, *d_coins_enc;
        CUDA_CHECK(cudaMalloc(&d_pk, pk_bytes));
        CUDA_CHECK(cudaMalloc(&d_sk, sk_bytes));
        CUDA_CHECK(cudaMalloc(&d_ct, ct_bytes));
        CUDA_CHECK(cudaMalloc(&d_ss, ss_bytes));
        CUDA_CHECK(cudaMalloc(&d_coins_kg, kg_bytes));
        CUDA_CHECK(cudaMalloc(&d_coins_enc, enc_bytes));
        CUDA_CHECK(cudaMemcpy(d_coins_kg, h_coins_kg, kg_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coins_enc, h_coins_enc, enc_bytes, cudaMemcpyHostToDevice));

        run_serial_kem_round(d_pk, d_sk, d_ct, d_ss, d_coins_kg, d_coins_enc, batch_count, n_ops);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_pk); cudaFree(d_sk); cudaFree(d_ct); cudaFree(d_ss);
        cudaFree(d_coins_kg); cudaFree(d_coins_enc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double alloc_each_ms = get_time_ms() - t0;

    uint8_t *d_pk, *d_sk, *d_ct, *d_ss, *d_coins_kg, *d_coins_enc;
    CUDA_CHECK(cudaMalloc(&d_pk, pk_bytes));
    CUDA_CHECK(cudaMalloc(&d_sk, sk_bytes));
    CUDA_CHECK(cudaMalloc(&d_ct, ct_bytes));
    CUDA_CHECK(cudaMalloc(&d_ss, ss_bytes));
    CUDA_CHECK(cudaMalloc(&d_coins_kg, kg_bytes));
    CUDA_CHECK(cudaMalloc(&d_coins_enc, enc_bytes));

    CUDA_CHECK(cudaDeviceSynchronize());
    t0 = get_time_ms();
    for (int r = 0; r < rounds; r++) {
        CUDA_CHECK(cudaMemcpy(d_coins_kg, h_coins_kg, kg_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coins_enc, h_coins_enc, enc_bytes, cudaMemcpyHostToDevice));
        run_serial_kem_round(d_pk, d_sk, d_ct, d_ss, d_coins_kg, d_coins_enc, batch_count, n_ops);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double reuse_ms = get_time_ms() - t0;

    double total_instances = (double)batch_count * (double)rounds * (double)n_ops;
    printf("  Alloc-each-round: total=%8.1f ms | per_round=%7.3f ms | full-kem throughput=%.0f instances/sec\n",
           alloc_each_ms, alloc_each_ms / rounds, total_instances * 1000.0 / alloc_each_ms);
    printf("  Reuse buffers:     total=%8.1f ms | per_round=%7.3f ms | full-kem throughput=%.0f instances/sec\n",
           reuse_ms, reuse_ms / rounds, total_instances * 1000.0 / reuse_ms);
    printf("  Reuse speedup:     %.3fx\n", alloc_each_ms / reuse_ms);

    cudaFree(d_pk); cudaFree(d_sk); cudaFree(d_ct); cudaFree(d_ss);
    cudaFree(d_coins_kg); cudaFree(d_coins_enc);
    free(h_coins_kg); free(h_coins_enc);
}

/* ================================================================
 *  Batch size 扫描
 * ================================================================ */
static void bench_batch_streams(int batch_count, int n_ops, int nstreams)
{
    printf("\n--- batch=%d n_ops=%d mode=serial streams=%d ---\n",
           batch_count, n_ops, nstreams);

    cudaStream_t *streams = (cudaStream_t *)calloc((size_t)nstreams, sizeof(cudaStream_t));
    uint8_t **d_pk = (uint8_t **)calloc((size_t)nstreams, sizeof(uint8_t *));
    uint8_t **d_sk = (uint8_t **)calloc((size_t)nstreams, sizeof(uint8_t *));
    uint8_t **d_ct = (uint8_t **)calloc((size_t)nstreams, sizeof(uint8_t *));
    uint8_t **d_ss = (uint8_t **)calloc((size_t)nstreams, sizeof(uint8_t *));
    uint8_t **d_coins_kg = (uint8_t **)calloc((size_t)nstreams, sizeof(uint8_t *));
    uint8_t **d_coins_enc = (uint8_t **)calloc((size_t)nstreams, sizeof(uint8_t *));
    if (!streams || !d_pk || !d_sk || !d_ct || !d_ss || !d_coins_kg || !d_coins_enc) {
        fprintf(stderr, "OOM\n");
        exit(1);
    }

    size_t kg_bytes = (size_t)batch_count * 2 * PARAM_SYMBYTES;
    size_t enc_bytes = (size_t)batch_count * PARAM_SYMBYTES;
    uint8_t *h_coins_kg = (uint8_t *)malloc(kg_bytes);
    uint8_t *h_coins_enc = (uint8_t *)malloc(enc_bytes);
    if (!h_coins_kg || !h_coins_enc) { fprintf(stderr, "OOM\n"); exit(1); }

    srand(5678);
    for (size_t i = 0; i < kg_bytes; i++) h_coins_kg[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < enc_bytes; i++) h_coins_enc[i] = (uint8_t)(rand() & 0xFF);

    for (int s = 0; s < nstreams; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&d_pk[s],  (size_t)batch_count * PARAM_PUBLICKEYBYTES));
        CUDA_CHECK(cudaMalloc(&d_sk[s],  (size_t)batch_count * PARAM_SECRETKEYBYTES));
        CUDA_CHECK(cudaMalloc(&d_ct[s],  (size_t)batch_count * PARAM_CIPHERTEXTBYTES));
        CUDA_CHECK(cudaMalloc(&d_ss[s],  (size_t)batch_count * PARAM_SSBYTES));
        CUDA_CHECK(cudaMalloc(&d_coins_kg[s],  kg_bytes));
        CUDA_CHECK(cudaMalloc(&d_coins_enc[s], enc_bytes));
        CUDA_CHECK(cudaMemcpyAsync(d_coins_kg[s], h_coins_kg, kg_bytes, cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_coins_enc[s], h_coins_enc, enc_bytes, cudaMemcpyHostToDevice, streams[s]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double total_ops = (double)batch_count * (double)nstreams;

    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = get_time_ms();
    int kg_tpb = KEM_KEYGEN_TPB;
    int kg_blocks = (batch_count + kg_tpb - 1) / kg_tpb;
    for (int op = 0; op < n_ops; op++)
        for (int s = 0; s < nstreams; s++)
            batch_kem_keypair_serial_kernel<<<kg_blocks, kg_tpb, 0, streams[s]>>>(
                d_pk[s], d_sk[s], d_coins_kg[s], batch_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    double t_kg = (get_time_ms() - t0) / n_ops;
    printf("  Keygen:  %7.1f ms/round -> %.0f ops/sec\n", t_kg, total_ops * 1000.0 / t_kg);

    t0 = get_time_ms();
    int enc_tpb = KEM_ENCAPS_TPB;
    int enc_blocks = (batch_count + enc_tpb - 1) / enc_tpb;
    for (int op = 0; op < n_ops; op++)
        for (int s = 0; s < nstreams; s++)
            batch_kem_encaps_serial_kernel<<<enc_blocks, enc_tpb, 0, streams[s]>>>(
                d_ct[s], d_ss[s], d_pk[s], d_coins_enc[s], batch_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    double t_enc = (get_time_ms() - t0) / n_ops;
    printf("  Encaps:  %7.1f ms/round -> %.0f ops/sec\n", t_enc, total_ops * 1000.0 / t_enc);

    t0 = get_time_ms();
    int dec_tpb = KEM_DECAPS_TPB;
    int dec_blocks = (batch_count + dec_tpb - 1) / dec_tpb;
    for (int op = 0; op < n_ops; op++)
        for (int s = 0; s < nstreams; s++)
            batch_kem_decaps_serial_kernel<<<dec_blocks, dec_tpb, 0, streams[s]>>>(
                d_ss[s], d_ct[s], d_sk[s], batch_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    double t_dec = (get_time_ms() - t0) / n_ops;
    printf("  Decaps:  %7.1f ms/round -> %.0f ops/sec\n", t_dec, total_ops * 1000.0 / t_dec);

    for (int s = 0; s < nstreams; s++) {
        cudaFree(d_pk[s]); cudaFree(d_sk[s]); cudaFree(d_ct[s]); cudaFree(d_ss[s]);
        cudaFree(d_coins_kg[s]); cudaFree(d_coins_enc[s]);
        cudaStreamDestroy(streams[s]);
    }
    free(h_coins_kg); free(h_coins_enc);
    free(streams); free(d_pk); free(d_sk); free(d_ct); free(d_ss); free(d_coins_kg); free(d_coins_enc);
}

static void bench_sweep(void)
{
    int sizes[] = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 };
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));
    printf("\n=== Batch size 扫描: %s ===\n", algo_name());
    for (int i = 0; i < n; i++) {
        bench_batch(sizes[i], 3, 0);
    }
}

static const char *arg_value(int argc, char **argv, const char *name)
{
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return NULL;
}

static int has_arg(int argc, char **argv, const char *name)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

static int read_file_all_host(const char *path, uint8_t **out, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    long n;
    uint8_t *buf;
    if (!f) {
        fprintf(stderr, "open failed: %s\n", path);
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    n = ftell(f);
    if (n < 0) {
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    buf = (uint8_t *)malloc((size_t)n + 1u);
    if (!buf) {
        fclose(f);
        return -1;
    }
    if ((size_t)n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    buf[n] = 0;
    *out = buf;
    *out_len = (size_t)n;
    return 0;
}

static int read_file_exact_host(const char *path, uint8_t *buf, size_t len)
{
    uint8_t *tmp = NULL;
    size_t n = 0;
    int rc = read_file_all_host(path, &tmp, &n);
    if (rc != 0) return rc;
    if (n != len) {
        fprintf(stderr, "size mismatch: %s expected %zu got %zu\n", path, len, n);
        free(tmp);
        return -1;
    }
    memcpy(buf, tmp, len);
    free(tmp);
    return 0;
}

static int write_file_all_host(const char *path, const uint8_t *buf, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "write open failed: %s\n", path);
        return -1;
    }
    if (len > 0 && fwrite(buf, 1, len, f) != len) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

static void fill_random_host(uint8_t *buf, size_t len)
{
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        size_t n = fread(buf, 1, len, f);
        fclose(f);
        if (n == len) return;
    }
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)(rand() & 0xff);
}

static void duplicate_record(uint8_t *dst, const uint8_t *src, size_t item_len, int batch_count)
{
    for (int i = 0; i < batch_count; i++) {
        memcpy(dst + (size_t)i * item_len, src, item_len);
    }
}

static int run_kem_api_mode(int argc, char **argv, int batch_count)
{
    const int do_keygen = has_arg(argc, argv, "--api-kem-keygen");
    const int do_encaps = has_arg(argc, argv, "--api-kem-encaps");
    const int do_decaps = has_arg(argc, argv, "--api-kem-decaps");
    if (!do_keygen && !do_encaps && !do_decaps) return 0;
    if ((do_keygen ? 1 : 0) + (do_encaps ? 1 : 0) + (do_decaps ? 1 : 0) != 1) {
        fprintf(stderr, "select exactly one KEM API mode\n");
        return 2;
    }
    if (batch_count < 1) batch_count = 1;

    const size_t pk_batch_bytes = (size_t)batch_count * PARAM_PUBLICKEYBYTES;
    const size_t sk_batch_bytes = (size_t)batch_count * PARAM_SECRETKEYBYTES;
    const size_t ct_batch_bytes = (size_t)batch_count * PARAM_CIPHERTEXTBYTES;
    const size_t ss_batch_bytes = (size_t)batch_count * PARAM_SSBYTES;
    const size_t kg_bytes = (size_t)batch_count * 2 * PARAM_SYMBYTES;
    const size_t enc_bytes = (size_t)batch_count * PARAM_SYMBYTES;

    if (do_keygen) {
        const char *pk_out = arg_value(argc, argv, "--pk-out");
        const char *sk_out = arg_value(argc, argv, "--sk-out");
        if (!pk_out || !sk_out) {
            fprintf(stderr, "--api-kem-keygen requires --pk-out and --sk-out\n");
            return 2;
        }

        uint8_t *h_pk = (uint8_t *)malloc(PARAM_PUBLICKEYBYTES);
        uint8_t *h_sk = (uint8_t *)malloc(PARAM_SECRETKEYBYTES);
        uint8_t *h_coins_kg = (uint8_t *)malloc(kg_bytes);
        uint8_t *d_pk = NULL, *d_sk = NULL, *d_coins_kg = NULL;
        if (!h_pk || !h_sk || !h_coins_kg) {
            fprintf(stderr, "KEM API malloc failed\n");
            free(h_pk); free(h_sk); free(h_coins_kg);
            return 2;
        }
        fill_random_host(h_coins_kg, kg_bytes);
        CUDA_CHECK(cudaMalloc(&d_pk, pk_batch_bytes));
        CUDA_CHECK(cudaMalloc(&d_sk, sk_batch_bytes));
        CUDA_CHECK(cudaMalloc(&d_coins_kg, kg_bytes));
        CUDA_CHECK(cudaMemcpy(d_coins_kg, h_coins_kg, kg_bytes, cudaMemcpyHostToDevice));
        int tpb = KEM_KEYGEN_TPB;
        int blocks = (batch_count + tpb - 1) / tpb;
        batch_kem_keypair_serial_kernel<<<blocks, tpb>>>(d_pk, d_sk, d_coins_kg, batch_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_pk, d_pk, PARAM_PUBLICKEYBYTES, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sk, d_sk, PARAM_SECRETKEYBYTES, cudaMemcpyDeviceToHost));
        int rc = 0;
        if (write_file_all_host(pk_out, h_pk, PARAM_PUBLICKEYBYTES) != 0 ||
            write_file_all_host(sk_out, h_sk, PARAM_SECRETKEYBYTES) != 0) rc = 2;
        cudaFree(d_pk); cudaFree(d_sk); cudaFree(d_coins_kg);
        free(h_pk); free(h_sk); free(h_coins_kg);
        if (rc == 0) printf("API KEM keygen PASS batch=%d pk=%u sk=%u\n", batch_count, PARAM_PUBLICKEYBYTES, PARAM_SECRETKEYBYTES);
        return rc == 0 ? 1 : rc;
    }

    if (do_encaps) {
        const char *pk_in = arg_value(argc, argv, "--pk-in");
        const char *ct_out = arg_value(argc, argv, "--ct-out");
        const char *ss_out = arg_value(argc, argv, "--ss-out");
        if (!pk_in || !ct_out || !ss_out) {
            fprintf(stderr, "--api-kem-encaps requires --pk-in, --ct-out, and --ss-out\n");
            return 2;
        }

        uint8_t *h_pk_one = (uint8_t *)malloc(PARAM_PUBLICKEYBYTES);
        uint8_t *h_pk = (uint8_t *)malloc(pk_batch_bytes);
        uint8_t *h_ct = (uint8_t *)malloc(PARAM_CIPHERTEXTBYTES);
        uint8_t *h_ss = (uint8_t *)malloc(PARAM_SSBYTES);
        uint8_t *h_coins_enc = (uint8_t *)malloc(enc_bytes);
        uint8_t *d_pk = NULL, *d_ct = NULL, *d_ss = NULL, *d_coins_enc = NULL;
        if (!h_pk_one || !h_pk || !h_ct || !h_ss || !h_coins_enc) {
            fprintf(stderr, "KEM API malloc failed\n");
            free(h_pk_one); free(h_pk); free(h_ct); free(h_ss); free(h_coins_enc);
            return 2;
        }
        if (read_file_exact_host(pk_in, h_pk_one, PARAM_PUBLICKEYBYTES) != 0) {
            free(h_pk_one); free(h_pk); free(h_ct); free(h_ss); free(h_coins_enc);
            return 2;
        }
        duplicate_record(h_pk, h_pk_one, PARAM_PUBLICKEYBYTES, batch_count);
        fill_random_host(h_coins_enc, enc_bytes);
        CUDA_CHECK(cudaMalloc(&d_pk, pk_batch_bytes));
        CUDA_CHECK(cudaMalloc(&d_ct, ct_batch_bytes));
        CUDA_CHECK(cudaMalloc(&d_ss, ss_batch_bytes));
        CUDA_CHECK(cudaMalloc(&d_coins_enc, enc_bytes));
        CUDA_CHECK(cudaMemcpy(d_pk, h_pk, pk_batch_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coins_enc, h_coins_enc, enc_bytes, cudaMemcpyHostToDevice));
        int tpb = KEM_ENCAPS_TPB;
        int blocks = (batch_count + tpb - 1) / tpb;
        batch_kem_encaps_serial_kernel<<<blocks, tpb>>>(d_ct, d_ss, d_pk, d_coins_enc, batch_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ct, d_ct, PARAM_CIPHERTEXTBYTES, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ss, d_ss, PARAM_SSBYTES, cudaMemcpyDeviceToHost));
        int rc = 0;
        if (write_file_all_host(ct_out, h_ct, PARAM_CIPHERTEXTBYTES) != 0 ||
            write_file_all_host(ss_out, h_ss, PARAM_SSBYTES) != 0) rc = 2;
        cudaFree(d_pk); cudaFree(d_ct); cudaFree(d_ss); cudaFree(d_coins_enc);
        free(h_pk_one); free(h_pk); free(h_ct); free(h_ss); free(h_coins_enc);
        if (rc == 0) printf("API KEM encaps PASS batch=%d ct=%u ss=%u\n", batch_count, PARAM_CIPHERTEXTBYTES, PARAM_SSBYTES);
        return rc == 0 ? 1 : rc;
    }

    if (do_decaps) {
        const char *sk_in = arg_value(argc, argv, "--sk-in");
        const char *ct_in = arg_value(argc, argv, "--ct-in");
        const char *ss_out = arg_value(argc, argv, "--ss-out");
        if (!sk_in || !ct_in || !ss_out) {
            fprintf(stderr, "--api-kem-decaps requires --sk-in, --ct-in, and --ss-out\n");
            return 2;
        }

        uint8_t *h_sk_one = (uint8_t *)malloc(PARAM_SECRETKEYBYTES);
        uint8_t *h_ct_one = (uint8_t *)malloc(PARAM_CIPHERTEXTBYTES);
        uint8_t *h_sk = (uint8_t *)malloc(sk_batch_bytes);
        uint8_t *h_ct = (uint8_t *)malloc(ct_batch_bytes);
        uint8_t *h_ss = (uint8_t *)malloc(PARAM_SSBYTES);
        uint8_t *d_sk = NULL, *d_ct = NULL, *d_ss = NULL;
        if (!h_sk_one || !h_ct_one || !h_sk || !h_ct || !h_ss) {
            fprintf(stderr, "KEM API malloc failed\n");
            free(h_sk_one); free(h_ct_one); free(h_sk); free(h_ct); free(h_ss);
            return 2;
        }
        if (read_file_exact_host(sk_in, h_sk_one, PARAM_SECRETKEYBYTES) != 0 ||
            read_file_exact_host(ct_in, h_ct_one, PARAM_CIPHERTEXTBYTES) != 0) {
            free(h_sk_one); free(h_ct_one); free(h_sk); free(h_ct); free(h_ss);
            return 2;
        }
        duplicate_record(h_sk, h_sk_one, PARAM_SECRETKEYBYTES, batch_count);
        duplicate_record(h_ct, h_ct_one, PARAM_CIPHERTEXTBYTES, batch_count);
        CUDA_CHECK(cudaMalloc(&d_sk, sk_batch_bytes));
        CUDA_CHECK(cudaMalloc(&d_ct, ct_batch_bytes));
        CUDA_CHECK(cudaMalloc(&d_ss, ss_batch_bytes));
        CUDA_CHECK(cudaMemcpy(d_sk, h_sk, sk_batch_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ct, h_ct, ct_batch_bytes, cudaMemcpyHostToDevice));
        int tpb = KEM_DECAPS_TPB;
        int blocks = (batch_count + tpb - 1) / tpb;
        batch_kem_decaps_serial_kernel<<<blocks, tpb>>>(d_ss, d_ct, d_sk, batch_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ss, d_ss, PARAM_SSBYTES, cudaMemcpyDeviceToHost));
        int rc = write_file_all_host(ss_out, h_ss, PARAM_SSBYTES) != 0 ? 2 : 0;
        cudaFree(d_sk); cudaFree(d_ct); cudaFree(d_ss);
        free(h_sk_one); free(h_ct_one); free(h_sk); free(h_ct); free(h_ss);
        if (rc == 0) printf("API KEM decaps PASS batch=%d ss=%u\n", batch_count, PARAM_SSBYTES);
        return rc == 0 ? 1 : rc;
    }

    return 0;
}

/* ================================================================
 *  主函数
 * ================================================================ */
int main(int argc, char **argv)
{
    /* 解析参数 */
    int batch_count  = 65536;
    int n_ops        = 5;
    int do_sweep     = 0;
    int run_pipeline = 0;
    int do_correctness = 1;
    int nstreams     = 1;
    int profile_pipeline = 0;
    int reuse_rounds = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
            batch_count = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-ops") == 0 && i + 1 < argc)
            n_ops = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sweep") == 0)
            do_sweep = 1;
        else if (strcmp(argv[i], "--serial-only") == 0)
            run_pipeline = 0;
        else if (strcmp(argv[i], "--pipeline") == 0)
            run_pipeline = 1;
        else if (strcmp(argv[i], "--no-correctness") == 0)
            do_correctness = 0;
        else if (strcmp(argv[i], "--streams") == 0 && i + 1 < argc)
            nstreams = atoi(argv[++i]);
        else if (strcmp(argv[i], "--profile-pipeline") == 0)
            profile_pipeline = 1;
        else if (strcmp(argv[i], "--reuse-bench") == 0 && i + 1 < argc)
            reuse_rounds = atoi(argv[++i]);
    }

    /* 打印设备信息 */
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        #if GPU_USE_HIP
        printf("GPU: %s (%s, %d CUs, %.1f GB VRAM)\n",
            prop.name,
            prop.gcnArchName,
            prop.multiProcessorCount,
            prop.totalGlobalMem / 1e9);
        #else
        printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB VRAM)\n",
            prop.name, prop.major, prop.minor,
            prop.multiProcessorCount,
            prop.totalGlobalMem / 1e9);
        #endif
        printf("Runtime: %s\n", GPU_RUNTIME_NAME);
    printf("Algorithm: %s  K=%d  Q=%d\n", algo_name(), PARAM_K, PARAM_Q);

    /* 设置 GPU 堆栈大小 (kem 函数需要 ~20KB 堆栈) */
    {
        cudaError_t se = cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);
        if (se != cudaSuccess) {
            fprintf(stderr, "Warning: cudaDeviceSetLimit(stack, 64KB) failed: %s\n",
                    cudaGetErrorString(se));
            cudaGetLastError();  /* 清除错误状态 */
        }
    }

    int api_rc = run_kem_api_mode(argc, argv, batch_count);
    if (api_rc != 0) return api_rc == 1 ? 0 : api_rc;

    /* 正确性测试 */
    if (do_correctness) {
        int ret = test_correctness();
        if (ret != 0) {
            fprintf(stderr, "正确性测试失败，中止性能测试\n");
            return ret;
        }
        printf("\n");
    }

    /* 吞吐量测试 */
    if (reuse_rounds > 0) {
        bench_reuse_buffers(batch_count, reuse_rounds, n_ops);
    } else if (do_sweep) {
        bench_sweep();
    } else {
        printf("=== 吞吐量测试: %s ===\n", algo_name());
        if (nstreams > 1)
            bench_batch_streams(batch_count, n_ops, nstreams);
        else
            bench_batch(batch_count, n_ops, 0, 0); /* serial mode default */
        if (run_pipeline) {
            /* 流水线模式 */
            bench_batch(batch_count, n_ops, 1, profile_pipeline);
        }
    }

    printf("\n完成.\n");
    return 0;
}
