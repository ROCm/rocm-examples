#include "hip/hip_runtime.h"
/*
 * main.cu — GPU 批量数字签名基准测试
 *
 * 支持 ML-DSA (44/65/87) 和 Aigis-sig (1/2/3) 共 6 种参数集
 * 流程:
 *   Phase 1: 单实例正确性验证 (随机测试向量, 输出全部输入/输出值)
 *   Phase 2: GPU批量 keygen / sign / verify 吞吐率测试 (输出 Instance 0 具体值)
 *
 * 批量数据采用 SoA (Structure-of-Arrays) 内存布局:
 *   pk_soa[byte_offset * N + instance_idx]
 *   precomp 批量: 单密钥对多实例 (soa_load/soa_store + 预计算分解 pipeline)
 *
 * 用法:
 *   exe [--batch N] [--sweep] [--quiet]
 *   --batch N   批次大小 (默认按参数集自动选择)
 *   --sweep     扫描多种批次大小: 64..32768
 *   --quiet     省略 Phase 1 的 hex 输出
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <time.h>
#include <errno.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include "config.h"
#include "params.h"
#include "sign.cuh"
#include "batch_ntt.cuh"
#include "batch_ops.cuh"
#include "batch_keygen.cuh"
#include "batch_verify.cuh"
#include "batch_sign.cuh"
#include "batch_sign_warp.cuh"

/* ================================================================
 *  常量定义
 * ================================================================ */
/* Conservative defaults plus larger Ada/4090 defaults. */
#if ALGORITHM == ALGO_MLDSA
  #if PARAM_MODE == 2    /* ML-DSA-44: pk=1312 sk=2560 sig=2420 ~6KB/inst */
    #define DEFAULT_BATCH 4096
    #define DEFAULT_BATCH_4090 16384
  #elif PARAM_MODE == 3  /* ML-DSA-65: pk=1952 sk=4032 sig=3309 ~9KB/inst */
    #define DEFAULT_BATCH 2048
    #define DEFAULT_BATCH_4090 32768
  #elif PARAM_MODE == 5  /* ML-DSA-87: pk=2592 sk=4896 sig=4627 ~12KB/inst */
    #define DEFAULT_BATCH 1024
    #define DEFAULT_BATCH_4090 16384
  #endif
#elif ALGORITHM == ALGO_AIGIS
  #if PARAM_MODE == 1    /* Aigis-1: pk=1056 sk=2448 sig=1852 ~5KB/inst */
    #define DEFAULT_BATCH 4096
    #define DEFAULT_BATCH_4090 16384
  #elif PARAM_MODE == 2  /* Aigis-2: pk=1312 sk=3376 sig=2445 ~7KB/inst */
    #define DEFAULT_BATCH 2048
    #define DEFAULT_BATCH_4090 16384
  #elif PARAM_MODE == 3  /* Aigis-3: pk=1568 sk=3888 sig=3046 ~8KB/inst */
    #define DEFAULT_BATCH 2048
    #define DEFAULT_BATCH_4090 16384
  #endif
#endif
#ifndef DEFAULT_BATCH_4090
#define DEFAULT_BATCH_4090 DEFAULT_BATCH
#endif
#ifndef CUDA_TARGET_ARCH
#define CUDA_TARGET_ARCH 0
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE      64
#endif
#define NUM_STREAMS     4

/* ================================================================
 *  命令行选项
 * ================================================================ */
typedef struct {
    int batch_size;
    int batch_auto;
    int sweep;
    int quiet;
    int throughput;
    int sample_only;
    int keygen_compare;
    int bench_paper;
    int bench_independent;
    int profile;
    int skip_keygen_oracle;
} Options;

static int g_profile = 0;
static int g_bench_independent = 0;

static int read_file_all(const char *path, uint8_t **out, size_t *out_len) {
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

static int read_file_exact_host(const char *path, uint8_t *buf, size_t len) {
    uint8_t *tmp = NULL;
    size_t n = 0;
    int rc = read_file_all(path, &tmp, &n);
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

static int write_file_all(const char *path, const uint8_t *buf, size_t len) {
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

static void fill_random_host(uint8_t *buf, size_t len) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        size_t n = fread(buf, 1, len, f);
        fclose(f);
        if (n == len) return;
    }
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)(rand() & 0xff);
}

#ifndef BATCH_KEYGEN_INTERNAL_MATERIAL
#define BATCH_KEYGEN_INTERNAL_MATERIAL 0
#endif

#ifndef BATCH_SIGN_PRECOMP_REUSE
#define BATCH_SIGN_PRECOMP_REUSE 0
#endif

#ifndef BATCH_SIGN_MONO_ENABLE
#define BATCH_SIGN_MONO_ENABLE 1
#endif

#ifndef BATCH_SIGN_DECOMP_ENABLE
#define BATCH_SIGN_DECOMP_ENABLE 1
#endif

#ifndef BATCH_SIGN_WARP_ENABLE
#define BATCH_SIGN_WARP_ENABLE 1
#endif

#ifndef BATCH_SIGN_WARP_PROFILE
#define BATCH_SIGN_WARP_PROFILE 0
#endif

#ifndef BATCH_SIGN_LARGE_STRATEGY_ENABLE
#define BATCH_SIGN_LARGE_STRATEGY_ENABLE 1
#endif

#ifndef BATCH_SIGN_LARGE_BATCH_THRESHOLD
#define BATCH_SIGN_LARGE_BATCH_THRESHOLD 4096
#endif

#ifndef BATCH_SIGN_NONCE_DIVERSIFY
#define BATCH_SIGN_NONCE_DIVERSIFY 0
#endif

#ifndef BATCH_SIGN_DECOMP_TAIL_ENABLE
#define BATCH_SIGN_DECOMP_TAIL_ENABLE 0
#endif

#ifndef BATCH_SIGN_CP_FUSE_ENABLE
#define BATCH_SIGN_CP_FUSE_ENABLE 0
#endif

#ifndef BATCH_SIGN_SAMPLE_DUP_YHAT
#define BATCH_SIGN_SAMPLE_DUP_YHAT 0
#endif

#ifndef BATCH_SIGN_DECOMP_CHECK_INTERVAL
#define BATCH_SIGN_DECOMP_CHECK_INTERVAL 4
#endif

#ifndef BATCH_SIGN_SAMPLE_TPB
#define BATCH_SIGN_SAMPLE_TPB 64
#endif

#ifndef BATCH_SIGN_HASH_TPB
#define BATCH_SIGN_HASH_TPB 32
#endif

#ifndef BATCH_SIGN_CHECK_TPB
#define BATCH_SIGN_CHECK_TPB 32
#endif

#ifndef BATCH_SIGN_DECOMP_ADAPTIVE_ENABLE
#define BATCH_SIGN_DECOMP_ADAPTIVE_ENABLE 0
#endif

static const char *keygen_ind_sample_mode_name(void) {
#if BATCH_KEYGEN_MATRIX_A_COOP
#if BATCH_KEYGEN_SECRET_ETA_COOP
    return "sample-coop-full";
#elif BATCH_KEYGEN_MATRIX_A_COOP_SUBWARP
    return "matrixA-coop-subwarp";
#else
    return "matrixA-coop-warp";
#endif
#elif BATCH_KEYGEN_MATRIX_A_LANEOPT
    return "matrixA-laneopt";
#elif BATCH_KEYGEN_SECRET_ETA_COOP
#if BATCH_KEYGEN_SECRET_ETA_AIGIS5_SPLIT
    return "eta2-aigis5-coop";
#else
    return "eta-coop";
#endif
#elif BATCH_KEYGEN_SAMPLE_SPLIT_FAST || BATCH_KEYGEN_MATRIX_A_FAST || BATCH_KEYGEN_SECRET_ETA_FAST
    return "split-baseline";
#else
    return "old-fused";
#endif
}

static const char *keygen_paper_sample_mode_name(void) {
#if BATCH_KEYGEN_SECRET_ETA_COOP
#if BATCH_KEYGEN_MATRIX_A_COOP
    return "sample-coop-full";
#elif BATCH_KEYGEN_SECRET_ETA_AIGIS5_SPLIT
    return "eta2-aigis5-coop";
#else
    return "eta-coop";
#endif
#elif BATCH_KEYGEN_SAMPLE_SPLIT_FAST || BATCH_KEYGEN_SECRET_ETA_FAST
    return "split-baseline";
#else
    return "old-fused";
#endif
}

static const char *internal_material_mode_name(void) {
#if BATCH_KEYGEN_INTERNAL_MATERIAL
    return "internal-material";
#else
    return "pk-sk-precompute";
#endif
}

static const char *sign_precomp_mode_name(void) {
#if BATCH_SIGN_DECOMP_ENABLE && !BATCH_SIGN_MONO_ENABLE && !BATCH_SIGN_PRECOMP_REUSE && !BATCH_SIGN_WARP_ENABLE
    return "sign-decomp-resource-aware";
#else
#if BATCH_SIGN_LARGE_STRATEGY_ENABLE
#if BATCH_SIGN_NONCE_DIVERSIFY
    return "large-batch-warp-strategy";
#else
    return "large-batch-mono-strategy";
#endif
#else
#if BATCH_SIGN_PRECOMP_REUSE
#if BATCH_SIGN_WARP_ENABLE
    return "sign-cache+warp-enabled";
#else
    return "sign-cache-enabled";
#endif
#else
#if BATCH_SIGN_MONO_ENABLE
#if BATCH_SIGN_WARP_ENABLE
    return "sign-mono+warp-enabled";
#else
    return "sign-mono-only";
#endif
#else
#if BATCH_SIGN_DECOMP_ENABLE
    return "sign-decomp-fallback";
#else
    return "sign-disabled";
#endif
#endif
#endif
#endif
#endif
}

static const char *policy_onoff(int enabled) {
    return enabled ? "on" : "off";
}

static BatchSignRuntimeOptions select_decomp_runtime_options(
    int batch,
    int independent_mode,
    const char **label)
{
    BatchSignRuntimeOptions opt = batch_sign_default_runtime_options();
    const char *name = "base";

#if BATCH_SIGN_DECOMP_ADAPTIVE_ENABLE
    opt.cp_fuse_enable = 0;
    opt.check_interval = 4;
    opt.hash_tpb = 32;
    opt.check_tpb = 32;

#if ALGORITHM == ALGO_MLDSA && PARAM_MODE == 2
    if (!independent_mode && batch >= 4096) {
        opt.check_interval = 16;
        name = "check16";
    }
#elif ALGORITHM == ALGO_MLDSA && PARAM_MODE == 5
    if (!independent_mode && batch <= 2048) {
        opt.check_interval = 8;
        name = "check8";
    }
#elif ALGORITHM == ALGO_AIGIS && PARAM_MODE == 2
    if (!independent_mode && batch <= 2048) {
        opt.check_interval = 16;
        name = "check16";
    } else if (!independent_mode && batch >= 4096) {
        name = "base";
    }
#endif
#endif

    if (label) *label = name;
    return opt;
}

static void print_rocm_sign_policy(int active_batch) {
    printf("ROCm sign policy: resource-aware hybrid candidates\n");
    printf("  decomp-pipeline=%s  monolithic-precomp=%s  cached-precomp=%s\n",
           policy_onoff(BATCH_SIGN_DECOMP_ENABLE),
           policy_onoff(BATCH_SIGN_MONO_ENABLE),
           policy_onoff(BATCH_SIGN_PRECOMP_REUSE));
    printf("  warp-path=%s  large-strategy=%s  threshold=%d  active_batch=%d\n",
           policy_onoff(BATCH_SIGN_MONO_ENABLE && BATCH_SIGN_WARP_ENABLE),
           policy_onoff(BATCH_SIGN_MONO_ENABLE && BATCH_SIGN_LARGE_STRATEGY_ENABLE),
           BATCH_SIGN_LARGE_BATCH_THRESHOLD,
           active_batch);
    printf("  decomp-cp-fuse=%s  decomp-tail=%s  yhat-copy-fuse=%s\n",
           policy_onoff(BATCH_SIGN_CP_FUSE_ENABLE),
           policy_onoff(BATCH_SIGN_DECOMP_TAIL_ENABLE),
           policy_onoff(BATCH_SIGN_SAMPLE_DUP_YHAT));
    printf("  decomp-adaptive=%s\n",
           policy_onoff(BATCH_SIGN_DECOMP_ADAPTIVE_ENABLE));
    printf("  decomp-check-interval=%d  ctrl-tpb(sample/hash/check)=%d/%d/%d\n",
           BATCH_SIGN_DECOMP_CHECK_INTERVAL,
           BATCH_SIGN_SAMPLE_TPB,
           BATCH_SIGN_HASH_TPB,
           BATCH_SIGN_CHECK_TPB);
#if defined(__HIP_PLATFORM_AMD__)
    printf("  backend=HIP/ROCm AMD; sign row records selected path label\n");
#else
    printf("  backend=HIP-compatible; sign row records selected path label\n");
#endif
#if BATCH_SIGN_DECOMP_ENABLE
    printf("  rationale=use decomp-pipeline to reduce monolithic private segment/scratch pressure\n");
#endif
}

static void print_usage(const char *prog) {
    printf("Usage: %s [--batch N] [--sweep] [--throughput] [--sample-only] [--keygen-compare] [--bench-paper] [--bench-independent] [--profile] [--skip-keygen-oracle] [--quiet]\n", prog);
    printf("  --batch N      batch size (auto default: conservative %d, RTX4090 %d)\n", DEFAULT_BATCH, DEFAULT_BATCH_4090);
    printf("  --sweep        sweep batch sizes: 64,128,256,512,1024,2048,4096,8192,16384,32768\n");
    printf("  --throughput   throughput scan: 256..32768, 10 runs avg, CSV output\n");
    printf("  --sample-only  sample-only microbench; skip NTT/matvec/pack/sign/verify\n");
    printf("  --keygen-compare  compare old vs active keygen path and exit; with --sample-only only compare sampling buffers\n");
    printf("  --bench-paper  paper-4090-style shared key/message/precompute benchmark (default)\n");
    printf("  --bench-independent  independent-real-batch mode; keygen seeds are independent\n");
    printf("  --profile      print lightweight pipeline/profile annotations\n");
    printf("  --skip-keygen-oracle  skip batch-vs-single keygen oracle check before profiling\n");
    printf("  --quiet        suppress Phase 1 hex dump\n");
}

static int parse_options(int argc, char **argv, Options *opt) {
    opt->batch_size = 0;
    opt->batch_auto = 1;
    opt->sweep = 0;
    opt->quiet = 0;
    opt->throughput = 0;
    opt->sample_only = 0;
    opt->keygen_compare = 0;
    opt->bench_paper = 1;
    opt->bench_independent = 0;
    opt->profile = 0;
    opt->skip_keygen_oracle = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            opt->batch_size = atoi(argv[++i]);
            if (opt->batch_size <= 0) { printf("Invalid batch size\n"); return -1; }
            opt->batch_auto = 0;
        } else if (strcmp(argv[i], "--sweep") == 0) {
            opt->sweep = 1;
        } else if (strcmp(argv[i], "--throughput") == 0) {
            opt->throughput = 1;
        } else if (strcmp(argv[i], "--sample-only") == 0) {
            opt->sample_only = 1;
        } else if (strcmp(argv[i], "--keygen-compare") == 0) {
            opt->keygen_compare = 1;
        } else if (strcmp(argv[i], "--bench-paper") == 0) {
            opt->bench_paper = 1;
            opt->bench_independent = 0;
        } else if (strcmp(argv[i], "--bench-independent") == 0) {
            opt->bench_paper = 0;
            opt->bench_independent = 1;
        } else if (strcmp(argv[i], "--profile") == 0) {
            opt->profile = 1;
        } else if (strcmp(argv[i], "--skip-keygen-oracle") == 0) {
            opt->skip_keygen_oracle = 1;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            opt->quiet = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 1;
        } else {
            printf("Unknown option: %s\n", argv[i]); return -1;
        }
    }
    g_profile = opt->profile;
    g_bench_independent = opt->bench_independent;
    return 0;
}

/* ================================================================
 *  CUDA 错误检查宏
 * ================================================================ */
#define CUDA_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        printf("CUDA error: %s (%s:%d)\n", \
               hipGetErrorString(_e), __FILE__, __LINE__); \
        rc = -1; goto cleanup; \
    } \
} while(0)

/* ================================================================
 *  SoA ↔ AoS 转换辅助函数
 *
 *  SoA 布局: soa_base[byte * N + idx]
 *  AoS 布局: aos_base[idx * item_bytes + byte]  (每线程本地连续缓冲)
 *
 *  soa_load:  从 SoA 全局内存加载到线程本地连续 buffer
 *  soa_store: 从线程本地连续 buffer 存回 SoA 全局内存
 * ================================================================ */
__device__ static void soa_load(uint8_t *local_buf, const uint8_t *soa_base,
                                 int idx, int N, int item_bytes) {
    for (int b = 0; b < item_bytes; ++b)
        local_buf[b] = soa_base[(size_t)b * N + idx];
}

__device__ static void soa_store(uint8_t *soa_base, const uint8_t *local_buf,
                                  int idx, int N, int item_bytes) {
    for (int b = 0; b < item_bytes; ++b)
        soa_base[(size_t)b * N + idx] = local_buf[b];
}

/* ================================================================
 *  GPU 内核
 * ================================================================ */

/* 单实例正确性测试: keygen + sign + verify + 篡改检测 */
__global__ void kernel_single_test(
    uint8_t *pk, uint8_t *sk, uint8_t *sig, size_t *siglen,
    const uint8_t *seed, const uint8_t *rnd,
    const uint8_t *msg, size_t mlen,
    const uint8_t *pre, size_t prelen,
    int *result)
{
    int r;
    r = crypto_sign_keypair(pk, sk, seed);
    if (r) { *result = -1; return; }

#if ALGORITHM == ALGO_MLDSA
    r = crypto_sign_signature(sig, siglen, msg, mlen, pre, prelen, rnd, sk);
#else
    r = crypto_sign_signature(sig, siglen, msg, mlen, rnd, sk);
#endif
    if (r) { *result = -2; return; }

#if ALGORITHM == ALGO_MLDSA
    r = crypto_sign_verify(sig, *siglen, msg, mlen, pre, prelen, pk);
#else
    r = crypto_sign_verify(sig, *siglen, msg, mlen, pk);
#endif
    if (r) { *result = -3; return; }

    /* 篡改 1 bit, 签名验证应失败 */
    sig[0] ^= 1;
#if ALGORITHM == ALGO_MLDSA
    r = crypto_sign_verify(sig, *siglen, msg, mlen, pre, prelen, pk);
#else
    r = crypto_sign_verify(sig, *siglen, msg, mlen, pk);
#endif
    sig[0] ^= 1;
    if (r == 0) { *result = -4; return; }

    *result = 0;
}

__global__ void kernel_keygen_only(uint8_t *pk, uint8_t *sk,
                                   const uint8_t *seed, int *result)
{
    int r = crypto_sign_keypair(pk, sk, seed);
    *result = r;
}

/* 设备端将 1 份 AoS 签名广播成 batch 份 AoS，避免 Host↔Device 往返 */
__global__ void kernel_cli_sign(uint8_t *sig, size_t *siglen, int *result,
                                const uint8_t *msg, size_t mlen,
                                const uint8_t *sk, const uint8_t *rnd)
{
#if ALGORITHM == ALGO_MLDSA
    const uint8_t *pre = msg;
    int r = crypto_sign_signature(sig, siglen, msg, mlen, pre, 0, rnd, sk);
#else
    int r = crypto_sign_signature(sig, siglen, msg, mlen, rnd, sk);
#endif
    *result = r;
}

__global__ void kernel_cli_verify(int *result,
                                  const uint8_t *sig, size_t siglen,
                                  const uint8_t *msg, size_t mlen,
                                  const uint8_t *pk)
{
#if ALGORITHM == ALGO_MLDSA
    const uint8_t *pre = msg;
    int r = crypto_sign_verify(sig, siglen, msg, mlen, pre, 0, pk);
#else
    int r = crypto_sign_verify(sig, siglen, msg, mlen, pk);
#endif
    *result = r;
}

__global__ void kernel_broadcast_sig_aos(uint8_t *dst, const uint8_t *src,
                                         int batch_count, int sig_bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * sig_bytes;
    if (idx < total) dst[idx] = src[idx % sig_bytes];
}

/* ================================================================
 *  预计算内核 — 同一密钥批量签名/验证
 * ================================================================ */

/* 单线程: 从 pk/sk 创建预计算数据 */
__global__ void kernel_create_precomp(precomp_t *pc,
                                       const uint8_t *pk, const uint8_t *sk) {
    create_precomp(pc, pk, sk);
}

/* 批量签名 (预计算): 每线程用共享预计算密钥签署消息 */
__global__ void __launch_bounds__(BLOCK_SIZE, 2)
kernel_batch_sign_precomp(
    uint8_t *sig_soa, size_t *siglen_arr,
    const uint8_t *msg, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *rnd,
    const precomp_t *pc,
    int *results, int N, int base_idx)
{
    int i = base_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint8_t sig_local[CRYPTO_BYTES];
#if BATCH_SIGN_NONCE_DIVERSIFY
    uint16_t nonce_start =
#if ALGORITHM == ALGO_AIGIS
        (uint16_t)(((unsigned int)i * PARAM_L) & 0xffffu);
#else
        (uint16_t)i;
#endif
#else
    uint16_t nonce_start = 0;
#endif
#if ALGORITHM == ALGO_MLDSA
    results[i] = crypto_sign_signature_precomp(
        sig_local, siglen_arr + i,
        msg, mlen, pre, prelen, rnd, pc, nonce_start);
#else
    results[i] = crypto_sign_signature_precomp(
        sig_local, siglen_arr + i, msg, mlen, rnd, pc, nonce_start);
#endif
    soa_store(sig_soa, sig_local, i, N, CRYPTO_BYTES);
}

typedef struct {
#if ALGORITHM == ALGO_MLDSA
    uint8_t mu[CRHBYTES];
    uint8_t rhoprime[CRHBYTES];
#else
    uint8_t mu[CRHBYTES];
    uint8_t key_mu[SEEDBYTES + CRHBYTES];
#endif
} sign_cache_t;

/* 单线程: paper-mode 共享消息/随机数时, 只派生一次签名哈希种子 */
__global__ void kernel_create_sign_cache(
    sign_cache_t *cache,
    const precomp_t *pc,
    const uint8_t *msg, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *rnd)
{
    keccak_state state;

#if ALGORITHM == ALGO_MLDSA
    shake256_init(&state);
    shake256_absorb(&state, pc->tr, TRBYTES);
    shake256_absorb(&state, pre, prelen);
    shake256_absorb(&state, msg, mlen);
    shake256_finalize(&state);
    shake256_squeeze(cache->mu, CRHBYTES, &state);

    shake256_init(&state);
    shake256_absorb(&state, pc->key, SEEDBYTES);
#if RNDBYTES > 0
    shake256_absorb(&state, rnd, RNDBYTES);
#endif
    shake256_absorb(&state, cache->mu, CRHBYTES);
    shake256_finalize(&state);
    shake256_squeeze(cache->rhoprime, CRHBYTES, &state);
#else
    shake256_init(&state);
    shake256_absorb(&state, pc->tr, TRBYTES);
    shake256_absorb(&state, msg, mlen);
    shake256_finalize(&state);
    shake256_squeeze(cache->mu, CRHBYTES, &state);

    memcpy(cache->key_mu, pc->key, SEEDBYTES);
    memcpy(cache->key_mu + SEEDBYTES, cache->mu, CRHBYTES);
#endif
}

/* 批量签名 (paper cached): 每线程复用共享 mu/rhoprime/key_mu, 仍独立执行 rejection loop */
__global__ void __launch_bounds__(BLOCK_SIZE, 2)
kernel_batch_sign_precomp_cached(
    uint8_t *sig_soa, size_t *siglen_arr,
    const sign_cache_t *cache,
    const precomp_t *pc,
    int *results, int N, int base_idx)
{
    int i = base_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint8_t sig_local[CRYPTO_BYTES];
#if BATCH_SIGN_NONCE_DIVERSIFY
    uint16_t nonce_start =
#if ALGORITHM == ALGO_AIGIS
        (uint16_t)(((unsigned int)i * PARAM_L) & 0xffffu);
#else
        (uint16_t)i;
#endif
#else
    uint16_t nonce_start = 0;
#endif
#if ALGORITHM == ALGO_MLDSA
    results[i] = crypto_sign_signature_precomp_cached(
        sig_local, siglen_arr + i, cache->mu, cache->rhoprime, pc, nonce_start);
#else
    results[i] = crypto_sign_signature_precomp_cached(
        sig_local, siglen_arr + i, cache->mu, cache->key_mu, pc, nonce_start);
#endif
    soa_store(sig_soa, sig_local, i, N, CRYPTO_BYTES);
}

/* 批量验证 (预计算): 每线程用共享预计算矩阵验证签名 */
__global__ void __launch_bounds__(BLOCK_SIZE, 2)
kernel_batch_verify_precomp(
    const uint8_t *sig_soa, const size_t *siglen_arr,
    const uint8_t *msg, size_t mlen,
    const uint8_t *pre, size_t prelen,
    const uint8_t *pk,
    const precomp_t *pc,
    int *results, int N, int base_idx)
{
    int i = base_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint8_t sig_local[CRYPTO_BYTES];
    soa_load(sig_local, sig_soa, i, N, CRYPTO_BYTES);
#if ALGORITHM == ALGO_MLDSA
    results[i] = crypto_sign_verify_precomp(
        sig_local, siglen_arr[i],
        msg, mlen, pre, prelen, pk, pc->mat);
#else
    results[i] = crypto_sign_verify_precomp(
        sig_local, siglen_arr[i],
        msg, mlen, pk, pc->mat);
#endif
}

/* ================================================================
 *  Host 辅助函数
 * ================================================================ */
static void print_hex(const char *label, const uint8_t *data, size_t len) {
    printf("%s (%zu bytes):\n", label, len);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
        if ((i + 1) % 32 == 0) printf("\n");
    }
    if (len % 32) printf("\n");
}

static int select_default_batch_for_device(void) {
    int devId = 0;
    hipDeviceProp_t prop;
    size_t free_mem = 0, total_mem = 0;
    if (hipGetDevice(&devId) != hipSuccess) return DEFAULT_BATCH;
    if (hipGetDeviceProperties(&prop, devId) != hipSuccess) return DEFAULT_BATCH;
    if (hipMemGetInfo(&free_mem, &total_mem) != hipSuccess) total_mem = 0;

    int runtime_sm = prop.major * 10 + prop.minor;
    if (runtime_sm >= 89 && total_mem >= (16ull * 1024ull * 1024ull * 1024ull)) {
        return DEFAULT_BATCH_4090;
    }
    return DEFAULT_BATCH;
}

static void print_info(int active_batch, int batch_auto) {
    int devId = 0;
    hipDeviceProp_t prop;
    hipGetDevice(&devId);
    hipGetDeviceProperties(&prop, devId);
    size_t free_mem = 0, total_mem = 0;
    hipMemGetInfo(&free_mem, &total_mem);
    printf("=== %s (Mode=%d) | Batch=%d%s ===\n",
           CRYPTO_ALGNAME, PARAM_MODE, active_batch,
           batch_auto ? " (auto)" : "");
    printf("GPU: %s  CC=%d.%d  SMs=%d  VRAM=%zuMB  L2=%dKB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           total_mem / (1024*1024), prop.l2CacheSize / 1024);
#if CUDA_TARGET_ARCH
    printf("Build: CUDA target=sm_%d  BLOCK_SIZE=%d\n", CUDA_TARGET_ARCH, BLOCK_SIZE);
    int runtime_sm = prop.major * 10 + prop.minor;
    if (runtime_sm != CUDA_TARGET_ARCH) {
        printf("Warning: binary was compiled for sm_%d but current GPU is sm_%d\n",
               CUDA_TARGET_ARCH, runtime_sm);
    }
#else
    printf("Build: CUDA target not recorded  BLOCK_SIZE=%d\n", BLOCK_SIZE);
#endif
    printf("Params: K=%d L=%d N=%d Q=%d  ETA=%d/%d  TAU=%d  GAMMA1=%d  OMEGA=%d\n",
           PARAM_K, PARAM_L, PARAM_N, PARAM_Q,
           PARAM_ETA_S1, PARAM_ETA_S2, PARAM_TAU, PARAM_GAMMA1, PARAM_OMEGA);
    printf("Sizes:  PK=%d  SK=%d  SIG=%d bytes\n\n",
           CRYPTO_PUBLICKEYBYTES, CRYPTO_SECRETKEYBYTES, CRYPTO_BYTES);
    print_rocm_sign_policy(active_batch);
    printf("\n");
}

/* 检查结果数组, 返回失败个数 */
static int count_failures(const int *h, int n) {
    int fails = 0;
    for (int i = 0; i < n; i++)
        if (h[i] != 0) fails++;
    return fails;
}

static int check_results(const int *h, int n, const char *stage) {
    int fails = 0, first_idx = -1, first_code = 0;
    for (int i = 0; i < n; i++) {
        if (h[i] != 0) {
            if (first_idx < 0) { first_idx = i; first_code = h[i]; }
            fails++;
        }
    }
    if (fails == 0) {
        printf("  [%s] correctness: all %d PASS\n", stage, n);
        return 0;
    }
    printf("  [%s] FAIL: %d/%d (first: idx=%d code=%d)\n",
           stage, fails, n, first_idx, first_code);
    return fails;
}

static double ops_from_ms(double count, float ms) {
    if (ms < 0.001f) return 0.0;
    return count * 1000.0 / (double)ms;
}

static int buffer_all_zero(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (data[i] != 0) return 0;
    }
    return 1;
}

static int check_host_key_material(const uint8_t *pk, const uint8_t *sk,
                                   const char *stage, int instance) {
    int pk_zero = buffer_all_zero(pk, CRYPTO_PUBLICKEYBYTES);
    int sk_zero = buffer_all_zero(sk, CRYPTO_SECRETKEYBYTES);
    if (pk_zero || sk_zero) {
        printf("[%s] FAIL: instance %d produced %s%s%s\n",
               stage, instance,
               pk_zero ? "all-zero PK" : "",
               (pk_zero && sk_zero) ? " and " : "",
               sk_zero ? "all-zero SK" : "");
        return -1;
    }
    return 0;
}

static int check_device_key_material_prefix(const uint8_t *d_pks,
                                            const uint8_t *d_sks,
                                            int batch_count,
                                            int check_count,
                                            const char *stage) {
    int n = check_count;
    if (n > batch_count) n = batch_count;
    if (n <= 0) return 0;

    size_t pk_bytes = (size_t)n * CRYPTO_PUBLICKEYBYTES;
    size_t sk_bytes = (size_t)n * CRYPTO_SECRETKEYBYTES;
    uint8_t *h_pk = (uint8_t *)malloc(pk_bytes);
    uint8_t *h_sk = (uint8_t *)malloc(sk_bytes);
    if (!h_pk || !h_sk) {
        printf("[%s] FAIL: host malloc failed during key material check\n", stage);
        free(h_pk);
        free(h_sk);
        return -1;
    }

    hipError_t err = hipMemcpy(h_pk, d_pks, pk_bytes, hipMemcpyDeviceToHost);
    if (err == hipSuccess) {
        err = hipMemcpy(h_sk, d_sks, sk_bytes, hipMemcpyDeviceToHost);
    }
    if (err != hipSuccess) {
        printf("[%s] FAIL: key material copy failed: %s\n",
               stage, hipGetErrorString(err));
        free(h_pk);
        free(h_sk);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        if (check_host_key_material(h_pk + (size_t)i * CRYPTO_PUBLICKEYBYTES,
                                    h_sk + (size_t)i * CRYPTO_SECRETKEYBYTES,
                                    stage, i) != 0) {
            free(h_pk);
            free(h_sk);
            return -1;
        }
    }

    free(h_pk);
    free(h_sk);
    return 0;
}

/* ================================================================
 *  Phase 1: 单实例正确性验证 — 输出全部输入/输出值
 * ================================================================ */
static int run_single_correctness(
    const uint8_t *h_seed, const uint8_t *h_rnd,
    const uint8_t *h_msg, size_t mlen,
    const uint8_t *h_ctx, size_t ctxlen,
    const uint8_t *h_pre, size_t prelen,
    int quiet)
{
    (void)h_seed;
    (void)h_rnd;
    (void)h_msg;
    (void)mlen;
    (void)h_ctx;
    (void)ctxlen;
    (void)h_pre;
    (void)prelen;
    (void)quiet;
    printf("=== Phase 1: Single-instance correctness skipped on AMD/HIP first-pass build ===\n\n");
    return 0;
}


static int run_keygen_oracle_check(const uint8_t *h_seed, int check_n, int quiet)
{
    (void)h_seed;
    (void)check_n;
    (void)quiet;
    return 0;
}


static int run_keygen_compare_batch(
    int N,
    const uint8_t *h_seed,
    int quiet,
    int sample_only)
{
    int rc = 0;
    unsigned char *d_base_seed = nullptr;
    KeygenCompareResult result;

    keygen_compare_result_clear(&result);

    if (!quiet) {
        printf("--- [Batch=%d] Keygen compare ---\n", N);
        printf("    mode=%s compare=%s\n",
               g_bench_independent ? "independent-real-batch" : "paper-4090-style",
               sample_only ? "sample-only" : "full-keygen");
        printf("    build: tr_hash_fixed=%d material=%s sign=%s sample_ind=%s sample_paper=%s\n",
               BATCH_KEYGEN_TR_HASH_FIXED,
               internal_material_mode_name(),
               sign_precomp_mode_name(),
               keygen_ind_sample_mode_name(),
               keygen_paper_sample_mode_name());
        fflush(stdout);
    }

    CUDA_CHECK(hipMalloc(&d_base_seed, SEEDBYTES));
    CUDA_CHECK(hipMemcpy(d_base_seed, h_seed, SEEDBYTES, hipMemcpyHostToDevice));

    rc = batch_keygen_compare_active_path(
        d_base_seed,
        N,
        g_bench_independent ? 0 : 1,
        sample_only,
        &result);

    if (rc == 0) {
        printf("[Keygen-compare] PASS: old vs active %s path matched for batch=%d\n\n",
               sample_only ? "sample" : "full",
               N);
    } else if (rc > 0) {
        printf("[Keygen-compare] first mismatch: stage=%s instance=%d byte_off=%zu elem_off=%zu ref=%lld cand=%lld\n\n",
               keygen_compare_stage_name(result.stage),
               result.instance,
               result.byte_offset,
               result.element_offset,
               (long long)result.ref_value,
               (long long)result.cand_value);
        rc = 1;
    } else {
        printf("[Keygen-compare] FAILED to run compare\n\n");
    }

cleanup:
    hipFree(d_base_seed);
    return rc;
}

/* ================================================================
 *  Phase 2: 分解式批量性能基准测试
 *
 *  优化原理:
 *    1. 流水线分解: keygen/verify 拆成 7-11 个专用 kernel
 *    2. 共享内存 NTT: 128 线程/poly, shared memory 蝶形
 *    3. 2D Grid 矩阵向量乘: dim3(batch, K), 每系数一线程
 *    4. 栈缩减: 采样 48KB, 运算 4KB → GPU 利用率 >50%
 *    5. 共享矩阵 A (verify): 所有实例共享一份
 *    6. 多次迭代取平均: WARMUP + BENCH_ITERS
 * ================================================================ */
#define WARMUP_ITERS       3
#define BENCH_ITERS        5
#define SAMPLE_ONLY_ITERS  3
#define THROUGHPUT_RUNS    10

static float median3f(float a, float b, float c)
{
    if (a > b) { float t = a; a = b; b = t; }
    if (b > c) { float t = b; b = c; c = t; }
    if (a > b) { float t = a; a = b; b = t; }
    return b;
}

static int run_batch(
    int N,
    const uint8_t *h_seed, const uint8_t *h_rnd,
    const uint8_t *h_msg, size_t mlen,
    const uint8_t *h_pre, size_t prelen,
    int quiet,
    int bench_iters,
    float *out_kg_ms,
    float *out_sg_ms,
    float *out_vf_ms)
{
    int rc = 0;
    float ms = 0, ms_keygen = 0, ms_sign = 0, ms_verify = 0;
    float ms_keygen_old = 0.0f, ms_keygen_ind = 0.0f, ms_keygen_paper = -1.0f;
    int verify_fails = 0;
    double kg_ops = 0, sg_ops = 0, vf_ops = 0;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;

    /* 共用设备缓冲区: 单公钥 pk_one / sk_one (用于 sign 和 verify) */
    uint8_t *d_pk_one = nullptr, *d_sk_one = nullptr;
    uint8_t *d_sigs_for_verify = nullptr;
    int verify_uses_batch_sigs = 0;
    const char *chosen_sign_label = "precomp-monolithic";
    const char *chosen_keygen_label = "independent-old";
    uint8_t *d_base_seed = nullptr, *d_shared_rho = nullptr;
    BatchKeygenBuffers kbuf;
    KeygenProfile ind_profile;
    KeygenProfile paper_profile;
    memset(&kbuf, 0, sizeof(kbuf));
    keygen_profile_clear(&ind_profile);
    keygen_profile_clear(&paper_profile);
#if BATCH_KEYGEN_INTERNAL_MATERIAL
    int keygen_mat_shared = 0;
#endif

    auto print_keygen_profile_line = [&](const char *label,
                                         float total_ms,
                                         const char *sample_mode,
                                         const KeygenProfile &profile,
                                         int include_shared_a) {
        double ops = ops_from_ms((double)N, total_ms);
        printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [",
               label, N, total_ms, ops);
        if (include_shared_a) {
            printf("sharedA %.3f ", profile.shared_a_ms);
        }
        printf("sample %s sample_total %.3f sample_launch_gap %.3f copy %.3f ntt %.3f matvec %.3f post %.3f p2r %.3f pack_outer %.3f material %.3f]",
               sample_mode,
               profile.sample_ms,
               profile.sample_launch_gap_ms,
               profile.copy_ms,
               profile.ntt_ms,
               profile.matvec_ms,
               profile.post_ms,
               profile.p2r_ms,
               profile.pack_ms,
               profile.material_ms);
        printf(" sample_active[seed_expand %.3f matrixA_active %.3f eta_active %.3f]",
               profile.seed_expand_ms,
               profile.matrix_a_sample_ms,
               profile.secret_eta_sample_ms);
        {
            float pack_gap = profile.pack_inner_ms - profile.pack_fused_ms -
                             profile.pack_body_ms - profile.tr_hash_ms;
            if (pack_gap < 0.0f) pack_gap = 0.0f;
            printf(" pack[inner %.3f fused %.3f body %.3f tr %.3f gap %.3f]",
                   profile.pack_inner_ms,
                   profile.pack_fused_ms,
                   profile.pack_body_ms,
                   profile.tr_hash_ms,
                   pack_gap);
        }
        if (profile.matrix_a_coop_lanes > 0 || profile.secret_eta_coop_lanes > 0) {
            printf(" coop_lanes[matA %d eta %d] coop_ms[matA %.3f eta %.3f]",
                   profile.matrix_a_coop_lanes,
                   profile.secret_eta_coop_lanes,
                   profile.matrix_a_coop_ms,
                   profile.secret_eta_coop_ms);
        }
        if (profile.pack_header_ms > 0.0f || profile.pack_t1_ms > 0.0f ||
            profile.pack_eta_ms > 0.0f || profile.pack_t0_ms > 0.0f) {
            printf(" split[hdr %.3f t1 %.3f eta %.3f t0 %.3f]",
                   profile.pack_header_ms,
                   profile.pack_t1_ms,
                   profile.pack_eta_ms,
                   profile.pack_t0_ms);
        }
        printf("\n");
    };

    if (!quiet) {
        printf("--- [Batch=%d] Warp-parallel-SoA pipeline ---\n", N);
        printf("    mode=%s%s\n",
               g_bench_independent ? "independent-real-batch" : "paper-4090-style",
               g_profile ? " profile=on" : "");
         printf("    build: tr_hash_fixed=%d material=%s sign=%s sample_ind=%s sample_paper=%s\n",
             BATCH_KEYGEN_TR_HASH_FIXED,
             internal_material_mode_name(),
             sign_precomp_mode_name(),
             keygen_ind_sample_mode_name(),
             keygen_paper_sample_mode_name());
        fflush(stdout);
    }

    CUDA_CHECK(hipEventCreate(&ev0));
    CUDA_CHECK(hipEventCreate(&ev1));

    CUDA_CHECK(hipMalloc(&d_pk_one, CRYPTO_PUBLICKEYBYTES));
    CUDA_CHECK(hipMalloc(&d_sk_one, CRYPTO_SECRETKEYBYTES));

    /* ================================================================
     * [2a] 分解式 Keygen
     *
     * Pipeline: sample → copy → NTT → matvec → reduce → INVNTT → add → pack
     * 每步使用最优 kernel 配置:
    *   sample:      2 threads/instance sub-warp (SHAKE-heavy, 低并行)
    *   pack:        32 threads/block, 融合 power2round
     *   NTT:         128 threads/block (shared-memory 蝶形)
     *   matvec:      dim3(B,K) × N threads (2D grid, 每系数一线程)
     *   元素运算:    256 threads/block
     * ================================================================ */
    {
        /* 采样 kernel 需要较大栈 (SHAKE 展开矩阵 A) */
        size_t kg_stack = 48u * 1024u;
        if (hipDeviceSetLimit(hipLimitStackSize, kg_stack) != hipSuccess) {
            hipGetLastError();
            kg_stack = 64u * 1024u;
            hipDeviceSetLimit(hipLimitStackSize, kg_stack);
            hipGetLastError();
        }

        if (batch_keygen_alloc(&kbuf, N) != 0) {
            printf("  [Keygen] batch_keygen_alloc FAILED\n");
            rc = -1; goto cleanup;
        }

        CUDA_CHECK(hipMalloc(&d_base_seed, SEEDBYTES));
        CUDA_CHECK(hipMalloc(&d_shared_rho, SEEDBYTES));
        CUDA_CHECK(hipMemcpy(d_base_seed, h_seed, SEEDBYTES, hipMemcpyHostToDevice));

        printf("  Operation       Batch    Time(ms)    Throughput\n");
        printf("  ---------       -----    --------    ----------\n");

        for (int w = 0; w < WARMUP_ITERS; w++) {
            if (batch_keygen_pipeline_warp_opt(kbuf.d_pks, kbuf.d_sks, d_base_seed, &kbuf, N, NULL, 0, 1) != 0) {
                rc = -1; goto cleanup;
            }
        }
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipEventRecord(ev0));
        for (int it = 0; it < bench_iters; it++) {
            if (batch_keygen_pipeline_warp_opt(kbuf.d_pks, kbuf.d_sks, d_base_seed, &kbuf, N, NULL, 0, 1) != 0) {
                rc = -1; goto cleanup;
            }
        }
        CUDA_CHECK(hipEventRecord(ev1));
        CUDA_CHECK(hipEventSynchronize(ev1));
        CUDA_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        ms_keygen_old = ms / bench_iters;
        printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [baseline]\n",
               "Keygen-old", N, ms_keygen_old, ops_from_ms((double)N, ms_keygen_old));

        for (int w = 0; w < WARMUP_ITERS; w++) {
            if (batch_keygen_pipeline_warp_opt(kbuf.d_pks, kbuf.d_sks, d_base_seed, &kbuf, N, NULL, 0, 1) != 0) {
                rc = -1; goto cleanup;
            }
        }
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipEventRecord(ev0));
        for (int it = 0; it < bench_iters; it++) {
            if (batch_keygen_pipeline_warp_opt(kbuf.d_pks, kbuf.d_sks, d_base_seed, &kbuf, N, NULL, 0, 1) != 0) {
                rc = -1; goto cleanup;
            }
        }
        CUDA_CHECK(hipEventRecord(ev1));
        CUDA_CHECK(hipEventSynchronize(ev1));
        CUDA_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        ms_keygen_ind = ms / bench_iters;
        if (g_profile) {
            if (batch_keygen_pipeline_warp_opt(kbuf.d_pks, kbuf.d_sks, d_base_seed, &kbuf, N, &ind_profile, 0, 1) != 0) {
                rc = -1; goto cleanup;
            }
            CUDA_CHECK(hipDeviceSynchronize());
            print_keygen_profile_line("Keygen-ind-x", ms_keygen_ind,
                                      keygen_ind_sample_mode_name(), ind_profile, 0);
        } else {
            printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [sample %s]\n",
                   "Keygen-ind-x", N, ms_keygen_ind, ops_from_ms((double)N, ms_keygen_ind),
                   keygen_ind_sample_mode_name());
        }

        for (int w = 0; w < WARMUP_ITERS; w++) {
            if (batch_keygen_create_shared_rho_a(&kbuf, d_shared_rho, d_base_seed) != 0 ||
                batch_keygen_pipeline_paper_shared_rho_a(
                    kbuf.d_pks, kbuf.d_sks, d_base_seed, d_shared_rho, &kbuf, N, NULL, 0, 1) != 0) {
                rc = -1; goto cleanup;
            }
        }
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipEventRecord(ev0));
        for (int it = 0; it < bench_iters; it++) {
            if (batch_keygen_create_shared_rho_a(&kbuf, d_shared_rho, d_base_seed) != 0 ||
                batch_keygen_pipeline_paper_shared_rho_a(
                    kbuf.d_pks, kbuf.d_sks, d_base_seed, d_shared_rho, &kbuf, N, NULL, 0, 1) != 0) {
                rc = -1; goto cleanup;
            }
        }
        CUDA_CHECK(hipEventRecord(ev1));
        CUDA_CHECK(hipEventSynchronize(ev1));
        CUDA_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        ms_keygen_paper = ms / bench_iters;
        if (g_profile) {
            if (batch_keygen_create_shared_rho_a(&kbuf, d_shared_rho, d_base_seed, &paper_profile) != 0 ||
                batch_keygen_pipeline_paper_shared_rho_a(
                    kbuf.d_pks, kbuf.d_sks, d_base_seed, d_shared_rho, &kbuf, N, &paper_profile, 0, 1) != 0) {
                rc = -1; goto cleanup;
            }
            CUDA_CHECK(hipDeviceSynchronize());
            print_keygen_profile_line("Keygen-paper", ms_keygen_paper,
                                      keygen_paper_sample_mode_name(), paper_profile, 1);
        } else {
            printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [sharedA %.3f sample %s]\n",
                   "Keygen-paper", N, ms_keygen_paper, ops_from_ms((double)N, ms_keygen_paper),
                   paper_profile.shared_a_ms, keygen_paper_sample_mode_name());
        }

        ms_keygen = ms_keygen_old;
        chosen_keygen_label = "independent-old";
        if (ms_keygen_ind > 0.0f && ms_keygen_ind < ms_keygen) {
            ms_keygen = ms_keygen_ind;
            chosen_keygen_label = "independent-opt";
        }
        if (!g_bench_independent && ms_keygen_paper > 0.0f && ms_keygen_paper < ms_keygen) {
            ms_keygen = ms_keygen_paper;
            chosen_keygen_label = "paper-shared-rhoA";
        }

        if (strcmp(chosen_keygen_label, "paper-shared-rhoA") == 0) {
#if BATCH_KEYGEN_INTERNAL_MATERIAL
            keygen_mat_shared = 1;
#endif
            batch_keygen_pipeline_paper_shared_rho_a(
                kbuf.d_pks, kbuf.d_sks, d_base_seed, d_shared_rho, &kbuf, N, NULL, 0, 1);
        } else if (strcmp(chosen_keygen_label, "independent-opt") == 0) {
#if BATCH_KEYGEN_INTERNAL_MATERIAL
            keygen_mat_shared = 0;
#endif
            batch_keygen_pipeline_warp_opt(kbuf.d_pks, kbuf.d_sks, d_base_seed, &kbuf, N, NULL, 0, 1);
        } else {
#if BATCH_KEYGEN_INTERNAL_MATERIAL
            keygen_mat_shared = 0;
#endif
            batch_keygen_pipeline_warp_opt(kbuf.d_pks, kbuf.d_sks, d_base_seed, &kbuf, N, NULL, 0, 1);
        }
        CUDA_CHECK(hipDeviceSynchronize());
        if (check_device_key_material_prefix(kbuf.d_pks, kbuf.d_sks,
                                             N, 8, "Keygen-selected") != 0) {
            rc = -1; goto cleanup;
        }

        /* 保存 instance[0] 的 pk/sk 供后续 sign+verify 使用 */
        CUDA_CHECK(hipMemcpy(d_pk_one, kbuf.d_pks,
                              CRYPTO_PUBLICKEYBYTES, hipMemcpyDeviceToDevice));
        CUDA_CHECK(hipMemcpy(d_sk_one, kbuf.d_sks,
                              CRYPTO_SECRETKEYBYTES, hipMemcpyDeviceToDevice));

        hipFree(d_shared_rho);
        hipFree(d_base_seed);
    }

    /* ================================================================
     * [2b] 预计算签名 (monolithic, 每线程独立)
     *
     * Sign 使用 rejection loop, 不易分解为 pipeline.
     * 使用共享密钥 precomp_t + 每线程独立签名.
     * ================================================================ */
    {
        /* 单线程创建预计算需要大栈 */
        size_t sign_precomp_stack = 128u * 1024u;
        hipDeviceSetLimit(hipLimitStackSize, sign_precomp_stack);
        hipGetLastError();

        precomp_t *d_pc = nullptr;
        CUDA_CHECK(hipMalloc(&d_pc, sizeof(precomp_t)));
    #if BATCH_KEYGEN_INTERNAL_MATERIAL
        batch_keygen_material_to_precomp_kernel<<<1, 1>>>(
            d_pc,
            kbuf.d_mat, kbuf.d_s1hat, kbuf.d_s2_ntt, kbuf.d_t0_ntt,
            kbuf.d_buf, kbuf.d_tr, 0, keygen_mat_shared);
    #else
        kernel_create_precomp<<<1, 1>>>(d_pc, d_pk_one, d_sk_one);
    #endif
        CUDA_CHECK(hipDeviceSynchronize());

        /* 签名使用较小栈 */
        size_t sign_stack = 64u * 1024u;
        hipDeviceSetLimit(hipLimitStackSize, sign_stack);
        hipGetLastError();

        /* 分配签名 SoA 缓冲区 */
        uint8_t *d_sig_soa = nullptr;
        size_t *d_siglen = nullptr;
        int *d_results = nullptr;
        int *h_results = nullptr;
        uint8_t *d_msg = nullptr, *d_rnd = nullptr, *d_pre_d = nullptr;
        size_t mem_sig = (size_t)N * CRYPTO_BYTES;

        h_results = (int *)calloc(N, sizeof(int));
        CUDA_CHECK(hipMalloc(&d_sig_soa, mem_sig));
        CUDA_CHECK(hipMalloc(&d_siglen, (size_t)N * sizeof(size_t)));
        CUDA_CHECK(hipMalloc(&d_results, (size_t)N * sizeof(int)));
        CUDA_CHECK(hipMalloc(&d_msg, mlen));
        CUDA_CHECK(hipMalloc(&d_rnd, RNDBYTES > 0 ? RNDBYTES : 1));
        CUDA_CHECK(hipMalloc(&d_pre_d, prelen > 0 ? prelen : 1));
        CUDA_CHECK(hipMemcpy(d_msg, h_msg, mlen, hipMemcpyHostToDevice));
#if RNDBYTES > 0
        CUDA_CHECK(hipMemcpy(d_rnd, h_rnd, RNDBYTES, hipMemcpyHostToDevice));
#endif
        if (prelen > 0)
            CUDA_CHECK(hipMemcpy(d_pre_d, h_pre, prelen, hipMemcpyHostToDevice));

        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sign_cache_t *d_sign_cache = nullptr;
        float ms_sign_cached = -1.0f;
        float ms_sign_mono = -1.0f;
        float ms_sign_warp = -1.0f;
        int cached_ok = 0;
        int mono_ok = 0;
        int warp_ok = 0;
        int sign_path = -1; /* 0=mono, 1=cached, 2=decomp, 3=warp, 4=warp-cached */
        unsigned long long *d_warp_stats = nullptr;
        unsigned long long h_warp_stats[WP_SIGN_STAT_COUNT];
        int warp_available = 0;
        size_t warp_smem = 0;

#if BATCH_SIGN_MONO_ENABLE && BATCH_SIGN_WARP_ENABLE
        warp_smem = batch_sign_warp_smem_bytes();
        {
            hipError_t we = batch_sign_warp_set_smem_attributes();
            if (we == hipSuccess) {
                warp_available = 1;
            } else {
                hipGetLastError();
                if (g_profile)
                    printf("  [Sign-warp] disabled: dynamic smem request %zu bytes/block rejected (%s)\n",
                           warp_smem, hipGetErrorString(we));
            }
        }
        if (warp_available && (g_profile || BATCH_SIGN_WARP_PROFILE)) {
            CUDA_CHECK(hipMalloc(&d_warp_stats,
                                  (size_t)WP_SIGN_STAT_COUNT * sizeof(unsigned long long)));
        }
#endif
        const int sign_large_batch =
            (BATCH_SIGN_LARGE_STRATEGY_ENABLE && N >= BATCH_SIGN_LARGE_BATCH_THRESHOLD);
        const int sign_real_nonce_batch =
#if BATCH_SIGN_NONCE_DIVERSIFY
            1;
#else
            0;
#endif
        const int prefer_warp_large =
            sign_large_batch && sign_real_nonce_batch && warp_available && !g_profile;

#if BATCH_SIGN_MONO_ENABLE && BATCH_SIGN_PRECOMP_REUSE
        if (!g_bench_independent) {
            CUDA_CHECK(hipMalloc(&d_sign_cache, sizeof(sign_cache_t)));
            kernel_create_sign_cache<<<1, 1>>>(
                d_sign_cache, d_pc, d_msg, mlen, d_pre_d, prelen, d_rnd);
            CUDA_CHECK(hipDeviceSynchronize());

            if (!prefer_warp_large) {
            for (int w = 0; w < WARMUP_ITERS; w++) {
                CUDA_CHECK(hipMemset(d_results, 0, N * sizeof(int)));
                kernel_batch_sign_precomp_cached<<<grid, BLOCK_SIZE>>>(
                    d_sig_soa, d_siglen, d_sign_cache, d_pc, d_results, N, 0);
                CUDA_CHECK(hipDeviceSynchronize());
            }

            CUDA_CHECK(hipEventRecord(ev0));
            for (int it = 0; it < bench_iters; it++) {
                CUDA_CHECK(hipMemset(d_results, 0, N * sizeof(int)));
                kernel_batch_sign_precomp_cached<<<grid, BLOCK_SIZE>>>(
                    d_sig_soa, d_siglen, d_sign_cache, d_pc, d_results, N, 0);
            }
            CUDA_CHECK(hipEventRecord(ev1));
            CUDA_CHECK(hipEventSynchronize(ev1));
            CUDA_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
            ms_sign_cached = ms / bench_iters;

            CUDA_CHECK(hipMemcpy(h_results, d_results, N * sizeof(int), hipMemcpyDeviceToHost));
            cached_ok = (count_failures(h_results, N) == 0);
            if (g_profile) {
                double cached_ops = ops_from_ms((double)N, ms_sign_cached);
                printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [%s]\n",
                       "Sign-cached", N, ms_sign_cached, cached_ops,
                       cached_ok ? "PASS" : "FAIL");
            }
            if (cached_ok) {
                ms_sign = ms_sign_cached;
                sign_path = 1;
                chosen_sign_label = "precomp-cached";
            }
            }
        }
#endif

        if (BATCH_SIGN_MONO_ENABLE && !prefer_warp_large) {
        for (int w = 0; w < WARMUP_ITERS; w++) {
            CUDA_CHECK(hipMemset(d_results, 0, N * sizeof(int)));
            kernel_batch_sign_precomp<<<grid, BLOCK_SIZE>>>(
                d_sig_soa, d_siglen, d_msg, mlen, d_pre_d, prelen,
                d_rnd, d_pc, d_results, N, 0);
            CUDA_CHECK(hipDeviceSynchronize());
        }

        CUDA_CHECK(hipEventRecord(ev0));
        for (int it = 0; it < bench_iters; it++) {
            CUDA_CHECK(hipMemset(d_results, 0, N * sizeof(int)));
            kernel_batch_sign_precomp<<<grid, BLOCK_SIZE>>>(
                d_sig_soa, d_siglen, d_msg, mlen, d_pre_d, prelen,
                d_rnd, d_pc, d_results, N, 0);
        }
        CUDA_CHECK(hipEventRecord(ev1));
        CUDA_CHECK(hipEventSynchronize(ev1));
        CUDA_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        ms_sign_mono = ms / bench_iters;
        if (g_profile) {
            double mono_ops = ops_from_ms((double)N, ms_sign_mono);
            printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [precomp-monolithic]\n",
                   "Sign-mono-old", N, ms_sign_mono, mono_ops);
        }
        CUDA_CHECK(hipMemcpy(h_results, d_results, N * sizeof(int), hipMemcpyDeviceToHost));
        mono_ok = (count_failures(h_results, N) == 0);
        if (g_profile)
            check_results(h_results, N, "Sign-mono-old");
        if (mono_ok && (sign_path < 0 || ms_sign_mono < ms_sign)) {
            ms_sign = ms_sign_mono;
            sign_path = 0;
            chosen_sign_label = "precomp-monolithic";
        }
        }

#if BATCH_SIGN_MONO_ENABLE && BATCH_SIGN_WARP_ENABLE
        {
            const int skip_warp_large_paper =
                sign_large_batch && !sign_real_nonce_batch && sign_path >= 0 && !g_profile;
        if (warp_available && !skip_warp_large_paper) {
            const int warp_cached =
#if BATCH_SIGN_PRECOMP_REUSE
                (!g_bench_independent && d_sign_cache != nullptr);
#else
                0;
#endif
            int grid_warp = (N + WP_SIGN_WARPS_BLOCK - 1) / WP_SIGN_WARPS_BLOCK;
            const char *warp_stage = warp_cached ? "Sign-warp-cached" : "Sign-warp";

            for (int w = 0; w < WARMUP_ITERS; w++) {
                CUDA_CHECK(hipMemset(d_results, 0, N * sizeof(int)));
                if (warp_cached) {
                    kernel_batch_sign_warp_precomp_cached<<<grid_warp, WP_SIGN_TPB, warp_smem>>>(
                        d_sig_soa, d_siglen, (const uint8_t *)d_sign_cache,
                        d_pc, d_results, N, 0, nullptr);
                } else {
                    kernel_batch_sign_warp_precomp<<<grid_warp, WP_SIGN_TPB, warp_smem>>>(
                        d_sig_soa, d_siglen, d_msg, mlen, d_pre_d, prelen,
                        d_rnd, d_pc, d_results, N, 0, nullptr);
                }
                CUDA_CHECK(hipGetLastError());
                CUDA_CHECK(hipDeviceSynchronize());
            }

            CUDA_CHECK(hipMemset(d_results, 0, N * sizeof(int)));
            if (d_warp_stats)
                CUDA_CHECK(hipMemset(d_warp_stats, 0,
                                      (size_t)WP_SIGN_STAT_COUNT * sizeof(unsigned long long)));

            CUDA_CHECK(hipEventRecord(ev0));
            for (int it = 0; it < bench_iters; it++) {
                CUDA_CHECK(hipMemset(d_results, 0, N * sizeof(int)));
                if (it == 0 && d_warp_stats)
                    CUDA_CHECK(hipMemset(d_warp_stats, 0,
                                          (size_t)WP_SIGN_STAT_COUNT * sizeof(unsigned long long)));
                if (warp_cached) {
                    kernel_batch_sign_warp_precomp_cached<<<grid_warp, WP_SIGN_TPB, warp_smem>>>(
                        d_sig_soa, d_siglen, (const uint8_t *)d_sign_cache,
                        d_pc, d_results, N, 0, (it == 0) ? d_warp_stats : nullptr);
                } else {
                    kernel_batch_sign_warp_precomp<<<grid_warp, WP_SIGN_TPB, warp_smem>>>(
                        d_sig_soa, d_siglen, d_msg, mlen, d_pre_d, prelen,
                        d_rnd, d_pc, d_results, N, 0, (it == 0) ? d_warp_stats : nullptr);
                }
                CUDA_CHECK(hipGetLastError());
            }
            CUDA_CHECK(hipEventRecord(ev1));
            CUDA_CHECK(hipEventSynchronize(ev1));
            CUDA_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
            ms_sign_warp = ms / bench_iters;

            CUDA_CHECK(hipMemcpy(h_results, d_results, N * sizeof(int), hipMemcpyDeviceToHost));
            warp_ok = (count_failures(h_results, N) == 0);
            if (g_profile) {
                double warp_ops = ops_from_ms((double)N, ms_sign_warp);
                printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [%s smem=%zu]\n",
                       warp_stage, N, ms_sign_warp, warp_ops,
                       warp_ok ? "PASS" : "FAIL", warp_smem);
                if (d_warp_stats) {
                    CUDA_CHECK(hipMemcpy(h_warp_stats, d_warp_stats,
                                          (size_t)WP_SIGN_STAT_COUNT * sizeof(unsigned long long),
                                          hipMemcpyDeviceToHost));
                    double avg_attempts = (N > 0)
                        ? (double)h_warp_stats[WP_SIGN_STAT_ATTEMPTS] / (double)N
                        : 0.0;
                    printf("  [%-14s] attempts=%.3f reject{s2=%llu z=%llu t0=%llu hint=%llu} ok=%llu\n",
                           warp_stage, avg_attempts,
                           h_warp_stats[WP_SIGN_STAT_REJ_S2],
                           h_warp_stats[WP_SIGN_STAT_REJ_Z],
                           h_warp_stats[WP_SIGN_STAT_REJ_T0],
                           h_warp_stats[WP_SIGN_STAT_REJ_HINT],
                           h_warp_stats[WP_SIGN_STAT_OK]);
                }
            }

            if (warp_ok && (sign_path < 0 || ms_sign_warp < ms_sign)) {
                if (d_sigs_for_verify) {
                    hipFree(d_sigs_for_verify);
                    d_sigs_for_verify = nullptr;
                }
                CUDA_CHECK(hipMalloc(&d_sigs_for_verify, (size_t)N * CRYPTO_BYTES));
                {
                    int total = N * CRYPTO_BYTES;
                    int tpb = 256;
                    int nblk = (total + tpb - 1) / tpb;
                    kernel_wp_sign_sig_soa_to_aos<<<nblk, tpb>>>(
                        d_sigs_for_verify, d_sig_soa, N);
                    CUDA_CHECK(hipGetLastError());
                    CUDA_CHECK(hipDeviceSynchronize());
                }
                verify_uses_batch_sigs = 1;
                ms_sign = ms_sign_warp;
                sign_path = warp_cached ? 4 : 3;
                chosen_sign_label = warp_cached ? "precomp-warp-cached" : "precomp-warp";
            }
        }
        }
#endif

        /* ----------------------------------------------------------------
         * [2b-decomp] 分解式批量签名 pipeline (算子级并行)
         *
         * 优化原理:
         *   · y 采样: 全批次并行 (per-instance 独立 SHAKE), 替代串行循环
         *   · NTT(y): shared-memory 批量 kernel (128 线程/poly, 复用 batch_ntt)
         *   · w = A·y: 共享矩阵 2D grid matvec (复用 batch_verify_matvec_kernel)
         *   · z/cs2/ct0: cp·shared_vec 批量 pointwise + INVNTT
         *   · 检查/提示/打包: per-instance 单线程 (小栈, 高并发)
         *   · 已完成实例通过 d_done 标志跳过, 减少拒绝轮尾部浪费
         * ---------------------------------------------------------------- */
        {
        const int skip_decomp_large_best =
            sign_large_batch && sign_path >= 0 && !g_profile;
        if ((g_profile || BATCH_SIGN_DECOMP_ENABLE) && !skip_decomp_large_best) {
            /* 分解式 pipeline 需要适中的栈 (check_pack kernel 最重, ~30KB/线程) */
            size_t decomp_stack = 64u * 1024u;
            hipDeviceSetLimit(hipLimitStackSize, decomp_stack);
            hipGetLastError();

            BatchSignPipeline bsp;
            memset(&bsp, 0, sizeof(bsp));
            if (batch_sign_alloc(&bsp, N) == 0) {
                const char *decomp_policy_label = nullptr;
                BatchSignRuntimeOptions decomp_runtime =
                    select_decomp_runtime_options(N, g_bench_independent, &decomp_policy_label);
                /* Warmup — 1 次 (含 rejection loop, 不计入时间) */
                int warm_rounds = 0, warm_done = 0;
                batch_sign_pipeline_ex(&bsp, N, d_pc, d_msg, mlen, d_pre_d, prelen,
                                       d_rnd, &decomp_runtime, &warm_rounds, &warm_done);

                /* Timed — BENCH_ITERS 次取平均 */
                float ms_sdp = 0;
                int last_rounds = 0, last_done = 0;
                CUDA_CHECK(hipEventRecord(ev0));
                for (int it = 0; it < bench_iters; it++)
                    batch_sign_pipeline_ex(&bsp, N, d_pc, d_msg, mlen, d_pre_d, prelen,
                                           d_rnd, &decomp_runtime, &last_rounds, &last_done);
                CUDA_CHECK(hipEventRecord(ev1));
                CUDA_CHECK(hipEventSynchronize(ev1));
                CUDA_CHECK(hipEventElapsedTime(&ms_sdp, ev0, ev1));
                ms_sdp /= bench_iters;
                if (g_profile) {
                    double decomp_ops = ops_from_ms((double)N, ms_sdp);
                    printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [policy=%s cp_fuse=%d check=%d ctrl=%d/%d rounds=%d done=%d]\n",
                           "Sign-decomp", N, ms_sdp, decomp_ops,
                           decomp_policy_label,
                           decomp_runtime.cp_fuse_enable,
                           decomp_runtime.check_interval,
                           decomp_runtime.hash_tpb,
                           decomp_runtime.check_tpb,
                           last_rounds, last_done);
                }

                /* 验证: 检查 d_done */
                int *h_dp_done = (int *)malloc(N * sizeof(int));
                hipMemcpy(h_dp_done, bsp.d_done, N * sizeof(int), hipMemcpyDeviceToHost);
                int dp_pass = 0;
                for (int i = 0; i < N; i++) if (h_dp_done[i]) dp_pass++;
                free(h_dp_done);

                if (g_profile) {
                    if (dp_pass == N)
                        printf("  [Sign-decomp] correctness: all %d PASS (last rounds=%d)\n",
                               dp_pass, last_rounds);
                    else
                        printf("  [Sign-decomp] WARN: only %d/%d completed\n", dp_pass, N);
                }

                if (dp_pass == N) {
                    if (sign_path < 0 || ms_sdp < ms_sign) {
                        ms_sign = ms_sdp;
                        if (d_sigs_for_verify) {
                            hipFree(d_sigs_for_verify);
                            d_sigs_for_verify = nullptr;
                        }
                        CUDA_CHECK(hipMalloc(&d_sigs_for_verify, (size_t)N * CRYPTO_BYTES));
                        CUDA_CHECK(hipMemcpy(d_sigs_for_verify, bsp.d_sigs,
                                              (size_t)N * CRYPTO_BYTES,
                                              hipMemcpyDeviceToDevice));
                        verify_uses_batch_sigs = 1;
                        sign_path = 2;
                        chosen_sign_label = (decomp_policy_label && strcmp(decomp_policy_label, "base") != 0)
                            ? "decomp-adaptive"
                            : "decomp-pipeline";
                    }
                }

                batch_sign_free(&bsp);
            } else {
                printf("  [Sign-decomp] alloc FAILED (out of VRAM)\n");
            }
        }
        }

        if (sign_path < 0) {
            printf("  [Sign] FAIL: no enabled signing path completed\n");
            rc = -1; goto cleanup;
        }
        if (!g_profile || quiet)
            printf("  [Sign] correctness: all %d PASS [%s]\n", N, chosen_sign_label);

        /* 为 verify 准备: decomp 成功时使用整批签名, 否则生成 1 个有效签名并广播 */
        uint8_t *d_sig_one = nullptr;
        if (!verify_uses_batch_sigs) {
            /* 用单线程签名生成 1 个有效签名 */
            size_t *d_siglen_one = nullptr;
            CUDA_CHECK(hipMalloc(&d_sig_one, CRYPTO_BYTES));
            CUDA_CHECK(hipMalloc(&d_siglen_one, sizeof(size_t)));
            CUDA_CHECK(hipMemset(d_sig_one, 0, CRYPTO_BYTES));

            size_t big_stack = 128u * 1024u;
            hipDeviceSetLimit(hipLimitStackSize, big_stack);
            hipGetLastError();

            /* 用预计算签名 kernel 生成 1 份签名 */
            int *d_vr = nullptr;
            CUDA_CHECK(hipMalloc(&d_vr, sizeof(int)));
            CUDA_CHECK(hipMemset(d_vr, 0, sizeof(int)));
            if (sign_path == 1) {
                kernel_batch_sign_precomp_cached<<<1, 1>>>(
                    d_sig_one, d_siglen_one, d_sign_cache, d_pc, d_vr, 1, 0);
            } else {
                kernel_batch_sign_precomp<<<1, 1>>>(
                    d_sig_one, d_siglen_one, d_msg, mlen, d_pre_d, prelen,
                    d_rnd, d_pc, d_vr, 1, 0);
            }
            CUDA_CHECK(hipDeviceSynchronize());
            hipFree(d_siglen_one); hipFree(d_vr);
        }

        free(h_results);
        hipFree(d_sig_soa); hipFree(d_siglen); hipFree(d_results);
        hipFree(d_msg); hipFree(d_rnd); hipFree(d_pre_d);
        hipFree(d_warp_stats);
        hipFree(d_sign_cache);
        hipFree(d_pc);

    /* ================================================================
     * [2c] 分解式 Verify
     *
     * Pipeline: precompute → unpack → chknorm → NTT(z) → matvec →
     *           challenge → NTT(cp) → sub_cp_t1 → reduce → INVNTT →
     *           normalize → use_hint → compare
     * 矩阵 A 和 t1_hat 所有实例共享 (只存一份)
     * ================================================================ */
    {
        /* 预计算需要大栈 (单线程) */
        size_t vc_stack = 128u * 1024u;
        hipDeviceSetLimit(hipLimitStackSize, vc_stack);
        hipGetLastError();

        BatchVerifyBuffers vbuf;
        memset(&vbuf, 0, sizeof(vbuf));
        if (batch_verify_alloc(&vbuf, N) != 0) {
            printf("  [Verify] batch_verify_alloc FAILED\n");
            hipFree(d_sig_one);
            hipFree(d_sigs_for_verify);
            rc = -1; goto cleanup;
        }

        if (verify_uses_batch_sigs) {
            CUDA_CHECK(hipMemcpy(vbuf.d_raw_sigs, d_sigs_for_verify,
                                  (size_t)N * CRYPTO_BYTES,
                                  hipMemcpyDeviceToDevice));
            hipFree(d_sigs_for_verify);
            d_sigs_for_verify = nullptr;
        } else {
            int total = N * CRYPTO_BYTES;
            int tpb = 256;
            int nblk = (total + tpb - 1) / tpb;
            kernel_broadcast_sig_aos<<<nblk, tpb>>>(
                vbuf.d_raw_sigs, d_sig_one, N, CRYPTO_BYTES);
            CUDA_CHECK(hipDeviceSynchronize());
            hipFree(d_sig_one);
            d_sig_one = nullptr;
        }

        /* 预计算: 直接复用 keygen 内部材料, 跳过 unpack_pk/matrix_expand/t1 NTT */
        {
#if BATCH_KEYGEN_INTERNAL_MATERIAL
            int total = PARAM_K * PARAM_L * PARAM_N;
            if (PARAM_K * PARAM_N > total) total = PARAM_K * PARAM_N;
            if (TRBYTES > total) total = TRBYTES;
            int tpb = 256;
            int nblk = (total + tpb - 1) / tpb;
            batch_keygen_material_to_verify_kernel<<<nblk, tpb>>>(
                vbuf.d_mat, vbuf.d_t1_hat, vbuf.d_tr,
                kbuf.d_mat, kbuf.d_t1_hat, kbuf.d_tr,
                0, keygen_mat_shared);
#else
            batch_verify_precompute_kernel<<<1, 1>>>(
                vbuf.d_mat, vbuf.d_t1_hat, vbuf.d_tr, d_pk_one);
#endif
        }
        CUDA_CHECK(hipDeviceSynchronize());

        /* Verify 分解 kernel 用小栈 */
        size_t verify_stack = 4u * 1024u;
        if (hipDeviceSetLimit(hipLimitStackSize, verify_stack) != hipSuccess) {
            hipGetLastError();
            verify_stack = 8u * 1024u;
            hipDeviceSetLimit(hipLimitStackSize, verify_stack);
            hipGetLastError();
        }

        /* 准备 per-instance 消息和 pre (所有实例用相同消息) */
        uint8_t *d_msgs_v = nullptr, *d_pre_v = nullptr;
        CUDA_CHECK(hipMalloc(&d_msgs_v, (size_t)N * mlen));
        CUDA_CHECK(hipMalloc(&d_pre_v, prelen > 0 ? prelen : 1));
        {
            uint8_t *h_msgs_v = (uint8_t *)malloc((size_t)N * mlen);
            for (int i = 0; i < N; i++)
                memcpy(h_msgs_v + (size_t)i * mlen, h_msg, mlen);
            CUDA_CHECK(hipMemcpy(d_msgs_v, h_msgs_v,
                                  (size_t)N * mlen, hipMemcpyHostToDevice));
            free(h_msgs_v);
        }
        if (prelen > 0)
            CUDA_CHECK(hipMemcpy(d_pre_v, h_pre, prelen, hipMemcpyHostToDevice));

        int *h_vresults = (int *)calloc(N, sizeof(int));

        /* Warmup */
        for (int w = 0; w < WARMUP_ITERS; w++) {
            batch_verify_pipeline_device_sigs(&vbuf, vbuf.d_raw_sigs, d_msgs_v, mlen,
                                              d_pre_v, prelen, N, h_vresults);
            hipDeviceSynchronize();
        }

        /* Timed (多次迭代取平均) */
        CUDA_CHECK(hipEventRecord(ev0));
        for (int it = 0; it < bench_iters; it++)
            batch_verify_pipeline_device_sigs(&vbuf, vbuf.d_raw_sigs, d_msgs_v, mlen,
                                              d_pre_v, prelen, N, h_vresults);
        CUDA_CHECK(hipEventRecord(ev1));
        CUDA_CHECK(hipEventSynchronize(ev1));
        CUDA_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        ms_verify = ms / bench_iters;
        {
            int vpass = 0;
            for (int i = 0; i < N; i++) if (h_vresults[i] == 0) vpass++;
            if (vpass < N) verify_fails = N - vpass;
        }

        free(h_vresults);
        hipFree(d_msgs_v); hipFree(d_pre_v);
        batch_verify_free(&vbuf);
    }
    } /* end sign scope (delayed close) */

    /* ---- 性能报告 (始终打印, --quiet 只抑制 Phase 1 hex dump) ---- */
    kg_ops = ops_from_ms((double)N, ms_keygen);
    sg_ops = ops_from_ms((double)N, ms_sign);
    vf_ops = ops_from_ms((double)N, ms_verify);
    printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [%s]\n",
           "Keygen",  N, ms_keygen, kg_ops, chosen_keygen_label);
    printf("  %-14s %8d  %10.3f ms  %12.0f ops/s  [%s]\n",
           "Sign",    N, ms_sign,   sg_ops, chosen_sign_label);
    printf("  %-14s %8d  %10.3f ms  %12.0f ops/s\n", "Verify",  N, ms_verify, vf_ops);
    if (verify_fails > 0) {
        printf("  [Verify] FAIL: %d/%d mismatched\n", verify_fails, N);
        printf("  [WARN] %d verify failures!\n", verify_fails);
        rc = -1;
    } else {
        printf("  [Verify] correctness: all %d PASS\n", N);
    }
    printf("\n");

    /* 通过输出参数返回计时数据 (供外部扫描模式使用) */
    if (out_kg_ms) *out_kg_ms = ms_keygen;
    if (out_sg_ms) *out_sg_ms = ms_sign;
    if (out_vf_ms) *out_vf_ms = ms_verify;

cleanup:
    if (ev0) hipEventDestroy(ev0);
    if (ev1) hipEventDestroy(ev1);
    hipFree(d_pk_one); hipFree(d_sk_one);
    hipFree(d_sigs_for_verify);
    batch_keygen_free(&kbuf);
    return rc;
}

static int run_sample_only_batch(
    int N,
    const uint8_t *h_seed,
    int quiet,
    int bench_iters)
{
    int rc = 0;
    BatchKeygenBuffers kbuf;
    unsigned char *d_base_seed = nullptr;
    unsigned char *d_shared_rho = nullptr;
    KeygenSampleOnlyProfile profile_sum;
    KeygenSampleOnlyProfile samples[SAMPLE_ONLY_ITERS];
    const int timed_iters = SAMPLE_ONLY_ITERS;
    const char *active_mode = g_bench_independent
        ? keygen_ind_sample_mode_name()
        : keygen_paper_sample_mode_name();
    const int print_active = strcmp(active_mode, "old-fused") != 0;

    memset(&kbuf, 0, sizeof(kbuf));
    keygen_sample_only_profile_clear(&profile_sum);

    if (!quiet) {
        printf("--- [Batch=%d] Sample-only microbench ---\n", N);
        printf("    mode=%s%s\n",
               g_bench_independent ? "independent-real-batch" : "paper-4090-style",
               g_profile ? " profile=on" : "");
        printf("    build: tr_hash_fixed=%d material=%s sign=%s sample_ind=%s sample_paper=%s\n",
               BATCH_KEYGEN_TR_HASH_FIXED,
               internal_material_mode_name(),
               sign_precomp_mode_name(),
               keygen_ind_sample_mode_name(),
               keygen_paper_sample_mode_name());
        fflush(stdout);
    }

    {
        size_t kg_stack = 48u * 1024u;
        if (hipDeviceSetLimit(hipLimitStackSize, kg_stack) != hipSuccess) {
            hipGetLastError();
            kg_stack = 64u * 1024u;
            hipDeviceSetLimit(hipLimitStackSize, kg_stack);
            hipGetLastError();
        }

        if (batch_keygen_alloc(&kbuf, N) != 0) {
            printf("  [Sample-only] batch_keygen_alloc FAILED\n");
            rc = -1; goto cleanup;
        }

        CUDA_CHECK(hipMalloc(&d_base_seed, SEEDBYTES));
        CUDA_CHECK(hipMemcpy(d_base_seed, h_seed, SEEDBYTES, hipMemcpyHostToDevice));
        if (!g_bench_independent)
            CUDA_CHECK(hipMalloc(&d_shared_rho, SEEDBYTES));

        for (int w = 0; w < WARMUP_ITERS; w++) {
            KeygenSampleOnlyProfile warmup;
            if (g_bench_independent) {
                if (batch_keygen_sample_only_independent(&kbuf, d_base_seed, N, &warmup) != 0) {
                    rc = -1; goto cleanup;
                }
            } else {
                if (batch_keygen_sample_only_paper(&kbuf, d_base_seed, d_shared_rho, N, &warmup) != 0) {
                    rc = -1; goto cleanup;
                }
            }
            CUDA_CHECK(hipDeviceSynchronize());
        }

        for (int it = 0; it < timed_iters; it++) {
            KeygenSampleOnlyProfile cur;
            if (g_bench_independent) {
                if (batch_keygen_sample_only_independent(&kbuf, d_base_seed, N, &cur) != 0) {
                    rc = -1; goto cleanup;
                }
            } else {
                if (batch_keygen_sample_only_paper(&kbuf, d_base_seed, d_shared_rho, N, &cur) != 0) {
                    rc = -1; goto cleanup;
                }
            }
                 samples[it] = cur;
        }
    }

            (void)bench_iters;
            profile_sum.old_fused_ms = median3f(samples[0].old_fused_ms,
                                 samples[1].old_fused_ms,
                                 samples[2].old_fused_ms);
            profile_sum.shared_a_ms = median3f(samples[0].shared_a_ms,
                                samples[1].shared_a_ms,
                                samples[2].shared_a_ms);
            profile_sum.split_seed_ms = median3f(samples[0].split_seed_ms,
                                  samples[1].split_seed_ms,
                                  samples[2].split_seed_ms);
            profile_sum.split_matrix_a_ms = median3f(samples[0].split_matrix_a_ms,
                                   samples[1].split_matrix_a_ms,
                                   samples[2].split_matrix_a_ms);
            profile_sum.split_eta_ms = median3f(samples[0].split_eta_ms,
                                 samples[1].split_eta_ms,
                                 samples[2].split_eta_ms);
            profile_sum.split_total_ms = median3f(samples[0].split_total_ms,
                                samples[1].split_total_ms,
                                samples[2].split_total_ms);
            profile_sum.split_launch_gap_ms = median3f(samples[0].split_launch_gap_ms,
                                     samples[1].split_launch_gap_ms,
                                     samples[2].split_launch_gap_ms);
            profile_sum.split_matrix_a_coop_ms = median3f(samples[0].split_matrix_a_coop_ms,
                                        samples[1].split_matrix_a_coop_ms,
                                        samples[2].split_matrix_a_coop_ms);
            profile_sum.split_eta_coop_ms = median3f(samples[0].split_eta_coop_ms,
                                   samples[1].split_eta_coop_ms,
                                   samples[2].split_eta_coop_ms);
            profile_sum.split_matrix_a_coop_lanes = samples[0].split_matrix_a_coop_lanes;
            profile_sum.split_eta_coop_lanes = samples[0].split_eta_coop_lanes;

            printf("  %-12s %-22s %8d  %10.3f ms  %12.0f ops/s\n",
                "Sample", "old-fused", N, profile_sum.old_fused_ms,
           ops_from_ms((double)N, profile_sum.old_fused_ms));
    if (profile_sum.shared_a_ms > 0.0f) {
             printf("  %-12s %-22s %8d  %10.3f ms  %12.0f ops/s\n",
                 "MatrixA", "sharedA", N, profile_sum.shared_a_ms,
               ops_from_ms(1.0, profile_sum.shared_a_ms));
    }
            if (print_active) {
                printf("  %-12s %-22s %8d  %10.3f ms  %12.0f ops/s  [seed %.3f matA %.3f eta %.3f total %.3f gap %.3f]",
                       "Sample", active_mode, N, profile_sum.split_total_ms,
                       ops_from_ms((double)N, profile_sum.split_total_ms),
                       profile_sum.split_seed_ms,
                       profile_sum.split_matrix_a_ms,
                       profile_sum.split_eta_ms,
                       profile_sum.split_total_ms,
                       profile_sum.split_launch_gap_ms);
                if (profile_sum.split_matrix_a_coop_lanes > 0 ||
                    profile_sum.split_eta_coop_lanes > 0) {
                    printf(" coop_lanes[matA %d eta %d] coop_ms[matA %.3f eta %.3f]",
                           profile_sum.split_matrix_a_coop_lanes,
                           profile_sum.split_eta_coop_lanes,
                           profile_sum.split_matrix_a_coop_ms,
                           profile_sum.split_eta_coop_ms);
                }
                printf("\n");
            }
    printf("\n");

cleanup:
    hipFree(d_shared_rho);
    hipFree(d_base_seed);
    batch_keygen_free(&kbuf);
    return rc;
}

/* ================================================================
 *  Phase 3: 批量吞吐量扫描 (--throughput)
 *
 *  自动循环 batch_size: 256,512,1024,2048,4096,8192,16384,32768
 *  每个 batch size 运行 THROUGHPUT_RUNS 次取平均, 输出 CSV 格式
 *  显存不足时跳过并记录 OOM
 * ================================================================ */
static void run_throughput_scan(
    const uint8_t *h_seed, const uint8_t *h_rnd,
    const uint8_t *h_msg, size_t mlen,
    const uint8_t *h_pre, size_t prelen)
{
    int batch_sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    int n_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    /* 创建 figure/ 目录 (保存架构图和 CSV) */
#ifdef _WIN32
    _mkdir("figure");
#else
    mkdir("figure", 0755);
#endif

    FILE *csv = fopen("figure/throughput.csv", "w");
    if (!csv) {
        printf("ERROR: cannot create figure/throughput.csv\n");
        return;
    }

    /* CSV 表头 */
    fprintf(csv, "batch_size,keygen_ms,keygen_ops_s,sign_ms,sign_ops_s,verify_ms,verify_ops_s,notes\n");

    printf("\n");
    printf("=== Batch Throughput Scan (avg of %d runs, CSV → figure/throughput.csv) ===\n",
           THROUGHPUT_RUNS);
    printf("%-10s %12s %14s %12s %14s %12s %14s\n",
           "Batch", "Kg(ms)", "Kg(ops/s)", "Sg(ms)", "Sg(ops/s)", "Vf(ms)", "Vf(ops/s)");
    printf("%-10s %12s %14s %12s %14s %12s %14s\n",
           "-----", "------", "----------", "------", "----------", "------", "----------");

    for (int i = 0; i < n_sizes; i++) {
        int N = batch_sizes[i];

        float kg = 0, sg = 0, vf = 0;
        int r = run_batch(N, h_seed, h_rnd, h_msg, mlen, h_pre, prelen,
                          1 /* quiet */, THROUGHPUT_RUNS /* 10 timed iters */,
                          &kg, &sg, &vf);

        if (r != 0) {
            printf("%-10d %12s %14s %12s %14s %12s %14s\n",
                   N, "FAIL", "FAIL", "FAIL", "FAIL", "FAIL", "FAIL");
            fprintf(csv, "%d,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL\n", N);
        } else {
            double kg_ops = ops_from_ms((double)N, kg);
            double sg_ops = ops_from_ms((double)N, sg);
            double vf_ops = ops_from_ms((double)N, vf);
            printf("%-10d %12.3f %14.0f %12.3f %14.0f %12.3f %14.0f\n",
                   N, kg, kg_ops, sg, sg_ops, vf, vf_ops);
            fprintf(csv, "%d,%.3f,%.0f,%.3f,%.0f,%.3f,%.0f,\n",
                    N, kg, kg_ops, sg, sg_ops, vf, vf_ops);
        }
        fflush(csv);
        fflush(stdout);
    }

    fclose(csv);
    printf("\n[throughput] CSV saved to figure/throughput.csv\n");
}

static const char *arg_value(int argc, char **argv, const char *name) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return NULL;
}

static int has_arg(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

static int run_cli_mode(int argc, char **argv) {
    int rc = 0;
    int h_result = 0;
    size_t h_siglen = 0;
    uint8_t *h_msg = NULL;
    size_t h_mlen = 0;
    uint8_t *h_pk = NULL, *h_sk = NULL, *h_sig = NULL, *h_seed = NULL, *h_rnd = NULL;
    uint8_t *d_pk = NULL, *d_sk = NULL, *d_sig = NULL, *d_msg = NULL, *d_seed = NULL, *d_rnd = NULL;
    size_t *d_siglen = NULL;
    int *d_result = NULL;

    const int do_keygen = has_arg(argc, argv, "--cli-keygen");
    const int do_sign = has_arg(argc, argv, "--cli-sign");
    const int do_verify = has_arg(argc, argv, "--cli-verify");
    if (!do_keygen && !do_sign && !do_verify) return 0;

    h_pk = (uint8_t *)calloc(1, CRYPTO_PUBLICKEYBYTES);
    h_sk = (uint8_t *)calloc(1, CRYPTO_SECRETKEYBYTES);
    h_sig = (uint8_t *)calloc(1, CRYPTO_BYTES);
    h_seed = (uint8_t *)calloc(1, SEEDBYTES);
#if RNDBYTES > 0
    h_rnd = (uint8_t *)calloc(1, RNDBYTES);
#else
    h_rnd = (uint8_t *)calloc(1, 1);
#endif
    if (!h_pk || !h_sk || !h_sig || !h_seed || !h_rnd) {
        fprintf(stderr, "CLI malloc failed\n");
        return 2;
    }

    CUDA_CHECK(hipMalloc(&d_pk, CRYPTO_PUBLICKEYBYTES));
    CUDA_CHECK(hipMalloc(&d_sk, CRYPTO_SECRETKEYBYTES));
    CUDA_CHECK(hipMalloc(&d_sig, CRYPTO_BYTES));
    CUDA_CHECK(hipMalloc(&d_seed, SEEDBYTES));
#if RNDBYTES > 0
    CUDA_CHECK(hipMalloc(&d_rnd, RNDBYTES));
#else
    CUDA_CHECK(hipMalloc(&d_rnd, 1));
#endif
    CUDA_CHECK(hipMalloc(&d_siglen, sizeof(size_t)));
    CUDA_CHECK(hipMalloc(&d_result, sizeof(int)));

    if (do_keygen) {
        const char *pk_out = arg_value(argc, argv, "--pk-out");
        const char *sk_out = arg_value(argc, argv, "--sk-out");
        const char *seed_in = arg_value(argc, argv, "--seed-in");
        if (!pk_out || !sk_out) {
            fprintf(stderr, "--cli-keygen requires --pk-out and --sk-out\n");
            rc = 2; goto cleanup;
        }
        if (seed_in) {
            if (read_file_exact_host(seed_in, h_seed, SEEDBYTES) != 0) { rc = 2; goto cleanup; }
        } else {
            fill_random_host(h_seed, SEEDBYTES);
        }
        CUDA_CHECK(hipMemcpy(d_seed, h_seed, SEEDBYTES, hipMemcpyHostToDevice));
        kernel_keygen_only<<<1, 1>>>(d_pk, d_sk, d_seed, d_result);
        CUDA_CHECK(hipGetLastError());
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
        if (h_result != 0) { fprintf(stderr, "CLI keygen failed: %d\n", h_result); rc = 3; goto cleanup; }
        CUDA_CHECK(hipMemcpy(h_pk, d_pk, CRYPTO_PUBLICKEYBYTES, hipMemcpyDeviceToHost));
        CUDA_CHECK(hipMemcpy(h_sk, d_sk, CRYPTO_SECRETKEYBYTES, hipMemcpyDeviceToHost));
        if (write_file_all(pk_out, h_pk, CRYPTO_PUBLICKEYBYTES) != 0 ||
            write_file_all(sk_out, h_sk, CRYPTO_SECRETKEYBYTES) != 0) { rc = 2; goto cleanup; }
        printf("CLI SIG keygen PASS pk=%d sk=%d\n", CRYPTO_PUBLICKEYBYTES, CRYPTO_SECRETKEYBYTES);
    } else if (do_sign) {
        const char *sk_in = arg_value(argc, argv, "--sk-in");
        const char *msg_in = arg_value(argc, argv, "--msg-in");
        const char *sig_out = arg_value(argc, argv, "--sig-out");
        const char *rnd_in = arg_value(argc, argv, "--rnd-in");
        if (!sk_in || !msg_in || !sig_out) {
            fprintf(stderr, "--cli-sign requires --sk-in, --msg-in, and --sig-out\n");
            rc = 2; goto cleanup;
        }
        if (read_file_exact_host(sk_in, h_sk, CRYPTO_SECRETKEYBYTES) != 0 ||
            read_file_all(msg_in, &h_msg, &h_mlen) != 0) { rc = 2; goto cleanup; }
#if RNDBYTES > 0
        if (rnd_in) {
            if (read_file_exact_host(rnd_in, h_rnd, RNDBYTES) != 0) { rc = 2; goto cleanup; }
        } else {
            fill_random_host(h_rnd, RNDBYTES);
        }
#endif
        CUDA_CHECK(hipMalloc(&d_msg, h_mlen > 0 ? h_mlen : 1));
        CUDA_CHECK(hipMemcpy(d_sk, h_sk, CRYPTO_SECRETKEYBYTES, hipMemcpyHostToDevice));
        if (h_mlen > 0) CUDA_CHECK(hipMemcpy(d_msg, h_msg, h_mlen, hipMemcpyHostToDevice));
#if RNDBYTES > 0
        CUDA_CHECK(hipMemcpy(d_rnd, h_rnd, RNDBYTES, hipMemcpyHostToDevice));
#endif
        kernel_cli_sign<<<1, 1>>>(d_sig, d_siglen, d_result, d_msg, h_mlen, d_sk, d_rnd);
        CUDA_CHECK(hipGetLastError());
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipMemcpy(&h_siglen, d_siglen, sizeof(size_t), hipMemcpyDeviceToHost));
        if (h_result != 0 || h_siglen != CRYPTO_BYTES) {
            fprintf(stderr, "CLI sign failed: result=%d siglen=%zu\n", h_result, h_siglen);
            rc = 3; goto cleanup;
        }
        CUDA_CHECK(hipMemcpy(h_sig, d_sig, CRYPTO_BYTES, hipMemcpyDeviceToHost));
        if (write_file_all(sig_out, h_sig, CRYPTO_BYTES) != 0) { rc = 2; goto cleanup; }
        printf("CLI SIG sign PASS sig=%d msg=%zu\n", CRYPTO_BYTES, h_mlen);
    } else if (do_verify) {
        const char *pk_in = arg_value(argc, argv, "--pk-in");
        const char *msg_in = arg_value(argc, argv, "--msg-in");
        const char *sig_in = arg_value(argc, argv, "--sig-in");
        if (!pk_in || !msg_in || !sig_in) {
            fprintf(stderr, "--cli-verify requires --pk-in, --msg-in, and --sig-in\n");
            rc = 2; goto cleanup;
        }
        if (read_file_exact_host(pk_in, h_pk, CRYPTO_PUBLICKEYBYTES) != 0 ||
            read_file_exact_host(sig_in, h_sig, CRYPTO_BYTES) != 0 ||
            read_file_all(msg_in, &h_msg, &h_mlen) != 0) { rc = 2; goto cleanup; }
        CUDA_CHECK(hipMalloc(&d_msg, h_mlen > 0 ? h_mlen : 1));
        CUDA_CHECK(hipMemcpy(d_pk, h_pk, CRYPTO_PUBLICKEYBYTES, hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_sig, h_sig, CRYPTO_BYTES, hipMemcpyHostToDevice));
        if (h_mlen > 0) CUDA_CHECK(hipMemcpy(d_msg, h_msg, h_mlen, hipMemcpyHostToDevice));
        kernel_cli_verify<<<1, 1>>>(d_result, d_sig, CRYPTO_BYTES, d_msg, h_mlen, d_pk);
        CUDA_CHECK(hipGetLastError());
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
        if (h_result == 0) {
            printf("CLI SIG verify PASS msg=%zu\n", h_mlen);
        } else {
            printf("CLI SIG verify FAIL code=%d msg=%zu\n", h_result, h_mlen);
            rc = 4; goto cleanup;
        }
    }

cleanup:
    hipFree(d_pk); hipFree(d_sk); hipFree(d_sig); hipFree(d_msg);
    hipFree(d_seed); hipFree(d_rnd); hipFree(d_siglen); hipFree(d_result);
    free(h_pk); free(h_sk); free(h_sig); free(h_seed); free(h_rnd); free(h_msg);
    return rc == 0 ? 1 : rc;
}

static void build_api_pre(uint8_t **h_pre, size_t *prelen) {
#if ALGORITHM == ALGO_MLDSA
    *prelen = 2;
    *h_pre = (uint8_t *)calloc(1, *prelen);
    if (*h_pre) {
        (*h_pre)[0] = 0;
        (*h_pre)[1] = 0;
    }
#else
    *prelen = 0;
    *h_pre = (uint8_t *)calloc(1, 1);
#endif
}

static void repeat_record(uint8_t *dst, const uint8_t *src, size_t item_len, int batch_count) {
    for (int i = 0; i < batch_count; i++) {
        memcpy(dst + (size_t)i * item_len, src, item_len);
    }
}

static int run_api_sig_sign(
    int batch_count,
    const char *msg_in,
    const char *pk_out,
    const char *sk_out,
    const char *sig_out)
{
    int rc = 0;
    uint8_t *h_msg = NULL, *h_pre = NULL, *h_seed = NULL, *h_rnd = NULL;
    uint8_t *h_pk = NULL, *h_sk = NULL, *h_sig = NULL;
    size_t h_mlen = 0, prelen = 0;
    uint8_t *d_pk_one = NULL, *d_sk_one = NULL;
    uint8_t *d_msg = NULL, *d_pre = NULL, *d_rnd = NULL, *d_base_seed = NULL;
    precomp_t *d_pc = NULL;
    BatchKeygenBuffers kbuf;
    BatchSignPipeline bsp;
    memset(&kbuf, 0, sizeof(kbuf));
    memset(&bsp, 0, sizeof(bsp));

    if (batch_count < 1) batch_count = 1;
    if (read_file_all(msg_in, &h_msg, &h_mlen) != 0) return 2;
    build_api_pre(&h_pre, &prelen);
    h_seed = (uint8_t *)malloc(SEEDBYTES);
    h_rnd = (uint8_t *)malloc(RNDBYTES > 0 ? RNDBYTES : 1);
    h_pk = (uint8_t *)malloc(CRYPTO_PUBLICKEYBYTES);
    h_sk = (uint8_t *)malloc(CRYPTO_SECRETKEYBYTES);
    h_sig = (uint8_t *)malloc(CRYPTO_BYTES);
    if (!h_pre || !h_seed || !h_rnd || !h_pk || !h_sk || !h_sig) {
        fprintf(stderr, "API SIG malloc failed\n");
        rc = 2; goto cleanup;
    }
    fill_random_host(h_seed, SEEDBYTES);
#if RNDBYTES > 0
    fill_random_host(h_rnd, RNDBYTES);
#endif

    if (batch_keygen_alloc(&kbuf, batch_count) != 0) {
        fprintf(stderr, "API SIG batch_keygen_alloc failed\n");
        rc = 3; goto cleanup;
    }
    CUDA_CHECK(hipMalloc(&d_base_seed, SEEDBYTES));
    CUDA_CHECK(hipMemcpy(d_base_seed, h_seed, SEEDBYTES, hipMemcpyHostToDevice));
    if (batch_keygen_pipeline_warp_opt(kbuf.d_pks, kbuf.d_sks, d_base_seed, &kbuf, batch_count, NULL, 0, 1) != 0) {
        fprintf(stderr, "API SIG batch keygen failed\n");
        rc = 3; goto cleanup;
    }
    CUDA_CHECK(hipDeviceSynchronize());
    CUDA_CHECK(hipMemcpy(h_pk, kbuf.d_pks, CRYPTO_PUBLICKEYBYTES, hipMemcpyDeviceToHost));
    CUDA_CHECK(hipMemcpy(h_sk, kbuf.d_sks, CRYPTO_SECRETKEYBYTES, hipMemcpyDeviceToHost));

    CUDA_CHECK(hipMalloc(&d_pk_one, CRYPTO_PUBLICKEYBYTES));
    CUDA_CHECK(hipMalloc(&d_sk_one, CRYPTO_SECRETKEYBYTES));
    CUDA_CHECK(hipMemcpy(d_pk_one, kbuf.d_pks, CRYPTO_PUBLICKEYBYTES, hipMemcpyDeviceToDevice));
    CUDA_CHECK(hipMemcpy(d_sk_one, kbuf.d_sks, CRYPTO_SECRETKEYBYTES, hipMemcpyDeviceToDevice));

    hipDeviceSetLimit(hipLimitStackSize, 128u * 1024u);
    hipGetLastError();
    CUDA_CHECK(hipMalloc(&d_pc, sizeof(precomp_t)));
    kernel_create_precomp<<<1, 1>>>(d_pc, d_pk_one, d_sk_one);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());

    CUDA_CHECK(hipMalloc(&d_msg, h_mlen > 0 ? h_mlen : 1));
    CUDA_CHECK(hipMalloc(&d_pre, prelen > 0 ? prelen : 1));
    CUDA_CHECK(hipMalloc(&d_rnd, RNDBYTES > 0 ? RNDBYTES : 1));
    if (h_mlen > 0) CUDA_CHECK(hipMemcpy(d_msg, h_msg, h_mlen, hipMemcpyHostToDevice));
    if (prelen > 0) CUDA_CHECK(hipMemcpy(d_pre, h_pre, prelen, hipMemcpyHostToDevice));
#if RNDBYTES > 0
    CUDA_CHECK(hipMemcpy(d_rnd, h_rnd, RNDBYTES, hipMemcpyHostToDevice));
#endif

    hipDeviceSetLimit(hipLimitStackSize, 64u * 1024u);
    hipGetLastError();
    if (batch_sign_alloc(&bsp, batch_count) != 0) {
        fprintf(stderr, "API SIG batch_sign_alloc failed\n");
        rc = 3; goto cleanup;
    }
    {
        const char *policy_label = NULL;
        int rounds = 0, done = 0;
        BatchSignRuntimeOptions runtime = select_decomp_runtime_options(batch_count, g_bench_independent, &policy_label);
        if (batch_sign_pipeline_ex(&bsp, batch_count, d_pc, d_msg, h_mlen, d_pre, prelen,
                                   d_rnd, &runtime, &rounds, &done) != 0 || done != batch_count) {
            fprintf(stderr, "API SIG decomp sign failed: done=%d/%d rounds=%d\n", done, batch_count, rounds);
            rc = 4; goto cleanup;
        }
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipMemcpy(h_sig, bsp.d_sigs, CRYPTO_BYTES, hipMemcpyDeviceToHost));
        printf("API SIG sign PASS batch=%d sig=%d policy=%s rounds=%d\n",
               batch_count, CRYPTO_BYTES, policy_label ? policy_label : "base", rounds);
    }

    if (write_file_all(pk_out, h_pk, CRYPTO_PUBLICKEYBYTES) != 0 ||
        write_file_all(sk_out, h_sk, CRYPTO_SECRETKEYBYTES) != 0 ||
        write_file_all(sig_out, h_sig, CRYPTO_BYTES) != 0) {
        rc = 2; goto cleanup;
    }

cleanup:
    hipFree(d_pk_one); hipFree(d_sk_one); hipFree(d_msg); hipFree(d_pre);
    hipFree(d_rnd); hipFree(d_base_seed); hipFree(d_pc);
    batch_keygen_free(&kbuf);
    batch_sign_free(&bsp);
    free(h_msg); free(h_pre); free(h_seed); free(h_rnd);
    free(h_pk); free(h_sk); free(h_sig);
    return rc == 0 ? 1 : rc;
}

static int run_api_sig_verify(
    int batch_count,
    const char *msg_in,
    const char *pk_in,
    const char *sig_in)
{
    int rc = 0;
    uint8_t *h_msg = NULL, *h_pre = NULL, *h_pk = NULL, *h_sig = NULL, *h_sigs = NULL, *h_msgs = NULL;
    size_t h_mlen = 0, prelen = 0;
    uint8_t *d_pk = NULL, *d_msgs = NULL, *d_pre = NULL;
    int *h_results = NULL;
    BatchVerifyBuffers vbuf;
    memset(&vbuf, 0, sizeof(vbuf));

    if (batch_count < 1) batch_count = 1;
    if (read_file_all(msg_in, &h_msg, &h_mlen) != 0) return 2;
    build_api_pre(&h_pre, &prelen);
    h_pk = (uint8_t *)malloc(CRYPTO_PUBLICKEYBYTES);
    h_sig = (uint8_t *)malloc(CRYPTO_BYTES);
    h_sigs = (uint8_t *)malloc((size_t)batch_count * CRYPTO_BYTES);
    h_msgs = (uint8_t *)malloc((size_t)batch_count * (h_mlen > 0 ? h_mlen : 1));
    h_results = (int *)calloc((size_t)batch_count, sizeof(int));
    if (!h_pre || !h_pk || !h_sig || !h_sigs || !h_msgs || !h_results) {
        fprintf(stderr, "API SIG verify malloc failed\n");
        rc = 2; goto cleanup;
    }
    if (read_file_exact_host(pk_in, h_pk, CRYPTO_PUBLICKEYBYTES) != 0 ||
        read_file_exact_host(sig_in, h_sig, CRYPTO_BYTES) != 0) {
        rc = 2; goto cleanup;
    }
    repeat_record(h_sigs, h_sig, CRYPTO_BYTES, batch_count);
    if (h_mlen > 0) repeat_record(h_msgs, h_msg, h_mlen, batch_count);

    hipDeviceSetLimit(hipLimitStackSize, 128u * 1024u);
    hipGetLastError();
    CUDA_CHECK(hipMalloc(&d_pk, CRYPTO_PUBLICKEYBYTES));
    CUDA_CHECK(hipMemcpy(d_pk, h_pk, CRYPTO_PUBLICKEYBYTES, hipMemcpyHostToDevice));
    if (batch_verify_alloc(&vbuf, batch_count) != 0) {
        fprintf(stderr, "API SIG batch_verify_alloc failed\n");
        rc = 3; goto cleanup;
    }
    batch_verify_precompute_kernel<<<1, 1>>>(vbuf.d_mat, vbuf.d_t1_hat, vbuf.d_tr, d_pk);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());

    hipDeviceSetLimit(hipLimitStackSize, 8u * 1024u);
    hipGetLastError();
    CUDA_CHECK(hipMalloc(&d_msgs, (size_t)batch_count * (h_mlen > 0 ? h_mlen : 1)));
    CUDA_CHECK(hipMalloc(&d_pre, prelen > 0 ? prelen : 1));
    if (h_mlen > 0) CUDA_CHECK(hipMemcpy(d_msgs, h_msgs, (size_t)batch_count * h_mlen, hipMemcpyHostToDevice));
    if (prelen > 0) CUDA_CHECK(hipMemcpy(d_pre, h_pre, prelen, hipMemcpyHostToDevice));
    if (batch_verify_pipeline(&vbuf, h_sigs, d_msgs, h_mlen, d_pre, prelen, batch_count, h_results) != 0) {
        fprintf(stderr, "API SIG verify pipeline failed\n");
        rc = 4; goto cleanup;
    }
    CUDA_CHECK(hipDeviceSynchronize());
    {
        int fails = count_failures(h_results, batch_count);
        if (fails == 0) {
            printf("API SIG verify PASS batch=%d sig=%d\n", batch_count, CRYPTO_BYTES);
        } else {
            printf("API SIG verify FAIL batch=%d fails=%d\n", batch_count, fails);
            rc = 5;
        }
    }

cleanup:
    hipFree(d_pk); hipFree(d_msgs); hipFree(d_pre);
    batch_verify_free(&vbuf);
    free(h_msg); free(h_pre); free(h_pk); free(h_sig); free(h_sigs); free(h_msgs); free(h_results);
    return rc == 0 ? 1 : rc;
}

static int run_sig_api_mode(int argc, char **argv) {
    const int do_sign = has_arg(argc, argv, "--api-sig-sign");
    const int do_verify = has_arg(argc, argv, "--api-sig-verify");
    if (!do_sign && !do_verify) return 0;
    if (do_sign && do_verify) {
        fprintf(stderr, "select exactly one SIG API mode\n");
        return 2;
    }
    int batch_count = 128;
    const char *batch_s = arg_value(argc, argv, "--batch");
    if (batch_s) batch_count = atoi(batch_s);

    if (do_sign) {
        const char *msg_in = arg_value(argc, argv, "--msg-in");
        const char *pk_out = arg_value(argc, argv, "--pk-out");
        const char *sk_out = arg_value(argc, argv, "--sk-out");
        const char *sig_out = arg_value(argc, argv, "--sig-out");
        if (!msg_in || !pk_out || !sk_out || !sig_out) {
            fprintf(stderr, "--api-sig-sign requires --msg-in, --pk-out, --sk-out, and --sig-out\n");
            return 2;
        }
        return run_api_sig_sign(batch_count, msg_in, pk_out, sk_out, sig_out);
    }

    if (do_verify) {
        const char *msg_in = arg_value(argc, argv, "--msg-in");
        const char *pk_in = arg_value(argc, argv, "--pk-in");
        const char *sig_in = arg_value(argc, argv, "--sig-in");
        if (!msg_in || !pk_in || !sig_in) {
            fprintf(stderr, "--api-sig-verify requires --msg-in, --pk-in, and --sig-in\n");
            return 2;
        }
        return run_api_sig_verify(batch_count, msg_in, pk_in, sig_in);
    }
    return 0;
}

/* ================================================================
 *  main
 * ================================================================ */
int main(int argc, char **argv) {
    int cli_rc = run_cli_mode(argc, argv);
    if (cli_rc != 0) return cli_rc == 1 ? 0 : cli_rc;
    int api_rc = run_sig_api_mode(argc, argv);
    if (api_rc != 0) return api_rc == 1 ? 0 : api_rc;
    Options opt;
    int r = parse_options(argc, argv, &opt);
    if (r > 0) return 0;
    if (r < 0) { print_usage(argv[0]); return 1; }

    if (opt.batch_auto) opt.batch_size = select_default_batch_for_device();
    print_info(opt.batch_size, opt.batch_auto);

    /* CUDA 栈空间 — 单线程正确性测试用较大栈 */
    {
        size_t phase1_stack = 128u * 1024u;
        if (hipDeviceSetLimit(hipLimitStackSize, phase1_stack) != hipSuccess) {
            hipGetLastError();
            printf("Warning: could not set CUDA stack size\n\n");
        }
    }

    /* 生成随机测试向量 */
    srand((unsigned)time(NULL));

    uint8_t h_seed[SEEDBYTES];
    for (int i = 0; i < SEEDBYTES; i++) h_seed[i] = (uint8_t)(rand() & 0xFF);

#if RNDBYTES > 0
    uint8_t h_rnd[RNDBYTES];
    for (int i = 0; i < RNDBYTES; i++) h_rnd[i] = (uint8_t)(rand() & 0xFF);
#else
    uint8_t h_rnd[1] = {0};
#endif

    size_t mlen = 32;
    uint8_t h_msg[32];
    for (size_t i = 0; i < mlen; i++) h_msg[i] = (uint8_t)(rand() & 0xFF);

#if ALGORITHM == ALGO_MLDSA
    size_t ctxlen = 32;
    uint8_t h_ctx[32];
    for (size_t i = 0; i < ctxlen; i++) h_ctx[i] = (uint8_t)(rand() & 0xFF);
#else
    size_t ctxlen = 0;
    uint8_t h_ctx[1] = {0};
#endif

    /* 构造 pre = (0, ctxlen, ctx) */
    size_t prelen = 2 + ctxlen;
    uint8_t h_pre[34];
    h_pre[0] = 0;
    h_pre[1] = (uint8_t)ctxlen;
    if (ctxlen > 0) memcpy(h_pre + 2, h_ctx, ctxlen);

    if (opt.sample_only) {
        if (opt.keygen_compare) {
            r = run_keygen_compare_batch(opt.batch_size, h_seed, opt.quiet, 1);
            return r != 0 ? 1 : 0;
        }

        printf("  %-12s %-22s %8s  %10s  %12s\n", "Operation", "Mode", "Batch", "Time(ms)", "Throughput");
        printf("  %-12s %-22s %8s  %10s  %12s\n", "---------", "----", "-----", "--------", "----------");
        fflush(stdout);

        if (opt.sweep) {
            int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
            int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
            for (int i = 0; i < nsizes; i++) {
                r = run_sample_only_batch(sizes[i], h_seed, opt.quiet, SAMPLE_ONLY_ITERS);
                if (r != 0) {
                    printf("Batch=%d FAILED, stopping sample-only sweep.\n", sizes[i]);
                    break;
                }
            }
        } else {
            r = run_sample_only_batch(opt.batch_size, h_seed, opt.quiet, SAMPLE_ONLY_ITERS);
        }
        return r != 0 ? 1 : 0;
    }

    if (opt.keygen_compare) {
        r = run_keygen_compare_batch(opt.batch_size, h_seed, opt.quiet, 0);
        return r != 0 ? 1 : 0;
    }

    /* 正确性验证 (单实例) */
    r = run_single_correctness(h_seed, h_rnd, h_msg, mlen,
                                h_ctx, ctxlen,
                                h_pre, prelen, opt.quiet);
    if (r != 0) {
        printf("Correctness FAILED.\n");
        return 1;
    }

    if (!opt.skip_keygen_oracle) {
        r = run_keygen_oracle_check(h_seed, 8, opt.quiet);
        if (r != 0) {
            printf("Keygen oracle check FAILED.\n");
            return 1;
        }
    } else if (!opt.quiet) {
        printf("[Keygen-oracle] skipped by --skip-keygen-oracle\n");
    }

    /* 批量性能基准 */
    if (opt.throughput) {
        run_throughput_scan(h_seed, h_rnd, h_msg, mlen, h_pre, prelen);
        return 0;
    }

    printf("  %-12s %8s  %10s  %12s\n", "Operation", "Batch", "Time(ms)", "Throughput");
    printf("  %-12s %8s  %10s  %12s\n", "---------", "-----", "--------", "----------");
    fflush(stdout);

    if (opt.sweep) {
        int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
        int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
        for (int i = 0; i < nsizes; i++) {
            r = run_batch(sizes[i], h_seed, h_rnd, h_msg, mlen,
                          h_pre, prelen, opt.quiet, BENCH_ITERS,
                          NULL, NULL, NULL);
            if (r != 0) {
                printf("Batch=%d FAILED, stopping sweep.\n", sizes[i]);
                break;
            }
        }
    } else {
        r = run_batch(opt.batch_size, h_seed, h_rnd, h_msg, mlen,
                      h_pre, prelen, opt.quiet, BENCH_ITERS,
                      NULL, NULL, NULL);
    }

    return r != 0 ? 1 : 0;
}
