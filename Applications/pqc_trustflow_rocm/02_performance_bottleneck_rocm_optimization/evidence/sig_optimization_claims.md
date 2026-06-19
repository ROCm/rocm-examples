# AMD ROCm Optimization Claims

## Implemented Candidates

- Stable signing remains the resource-aware `decomp-pipeline` path.
- `adaptive` is a runtime policy candidate: one binary selects the measured local winner by target, benchmark mode, and batch size, while falling back to base on cells where the matrix shows regressions.
- `check8` and `check16` measure whether fewer host-side done-count checks reduce ROCm synchronization overhead.
- `wave64_ctrl` measures whether 64-thread hash/check control kernels behave better on AMD wave64 hardware than 32-thread control kernels.
- `BATCH_SIGN_CP_FUSE_ENABLE` is implemented as a measured AMD candidate: one ROCm kernel computes `cp*s1`, `cp*s2`, and `cp*t0` products for each rejection round.
- `tail16_base` and `tail16_cp_fuse` separate small-tail finish behavior from the fused pointwise candidate.
- `yhat_dup` measures whether duplicating `y` at sample time beats the explicit device-to-device copy.
- The default build keeps these candidates off until the matrix proves a conservative target-specific gain.

## Current Large-Sweep Sign Best

| target | mode | batch | sign ops/s | path | log |
| --- | --- | ---: | ---: | --- | --- |
| aigis1 | independent | 16384 | 77989 | decomp-pipeline | aigis1_amd_independent_b16384.log |
| aigis1 | paper | 16384 | 83181 | decomp-pipeline | aigis1_amd_paper_b16384.log |
| aigis2 | independent | 32768 | 47248 | decomp-pipeline | aigis2_amd_independent_b32768.log |
| aigis2 | paper | 32768 | 50046 | decomp-pipeline | aigis2_amd_paper_b32768.log |
| aigis3 | independent | 32768 | 41240 | decomp-pipeline | aigis3_amd_independent_b32768.log |
| aigis3 | paper | 16384 | 41424 | decomp-pipeline | aigis3_amd_paper_b16384.log |
| mldsa44 | independent | 16384 | 106523 | decomp-pipeline | mldsa44_amd_independent_b16384.log |
| mldsa44 | paper | 16384 | 97354 | decomp-pipeline | mldsa44_amd_paper_b16384.log |
| mldsa65 | independent | 16384 | 70159 | decomp-pipeline | mldsa65_amd_independent_b16384.log |
| mldsa65 | paper | 8192 | 53968 | decomp-pipeline | mldsa65_amd_paper_b8192.log |
| mldsa87 | independent | 8192 | 49175 | decomp-pipeline | mldsa87_amd_independent_b8192.log |
| mldsa87 | paper | 8192 | 51185 | decomp-pipeline | mldsa87_amd_paper_b8192.log |

## AMD Feature Matrix Winners

| target | mode | batch | best variant | speedup vs base | sign ops/s | log |
| --- | --- | ---: | --- | ---: | ---: | --- |
| aigis1 | independent | 1024 | base | 1.0000 | 21166 | aigis1_base_independent_b1024_r1.log;aigis1_base_independent_b1024_r2.log |
| aigis1 | independent | 16384 | wave64_ctrl | 1.2076 | 86532 | aigis1_wave64_ctrl_independent_b16384_r1.log;aigis1_wave64_ctrl_independent_b16384_r2.log |
| aigis1 | independent | 32768 | wave64_ctrl | 1.0720 | 71926 | aigis1_wave64_ctrl_independent_b32768_r1.log;aigis1_wave64_ctrl_independent_b32768_r2.log |
| aigis1 | independent | 8192 | cp_fuse | 1.2375 | 64504 | aigis1_cp_fuse_independent_b8192_r1.log;aigis1_cp_fuse_independent_b8192_r2.log |
| aigis1 | paper | 1024 | wave64_ctrl | 1.4385 | 21375 | aigis1_wave64_ctrl_paper_b1024_r1.log;aigis1_wave64_ctrl_paper_b1024_r2.log |
| aigis1 | paper | 16384 | wave64_ctrl | 1.2519 | 82738 | aigis1_wave64_ctrl_paper_b16384_r1.log;aigis1_wave64_ctrl_paper_b16384_r2.log |
| aigis1 | paper | 32768 | tail16_base | 1.0767 | 72992 | aigis1_tail16_base_paper_b32768_r1.log;aigis1_tail16_base_paper_b32768_r2.log |
| aigis1 | paper | 8192 | adaptive | 1.0853 | 63246 | aigis1_adaptive_paper_b8192_r1.log;aigis1_adaptive_paper_b8192_r2.log |
| aigis2 | independent | 1024 | yhat_dup | 1.2031 | 12854 | aigis2_yhat_dup_independent_b1024.log;aigis2_yhat_dup_independent_b1024_r1.log;aigis2_yhat_dup_independent_b1024_r2.log |
| aigis2 | independent | 16384 | adaptive | 1.1586 | 50797 | aigis2_adaptive_independent_b16384.log;aigis2_adaptive_independent_b16384_r1.log;aigis2_adaptive_independent_b16384_r2.log |
| aigis2 | independent | 32768 | base | 1.0000 | 46575 | aigis2_base_independent_b32768.log;aigis2_base_independent_b32768_r1.log;aigis2_base_independent_b32768_r2.log |
| aigis2 | independent | 8192 | wave64_ctrl | 1.1985 | 41570 | aigis2_wave64_ctrl_independent_b8192.log;aigis2_wave64_ctrl_independent_b8192_r1.log;aigis2_wave64_ctrl_independent_b8192_r2.log |
| aigis2 | paper | 1024 | check8 | 1.3242 | 14708 | aigis2_check8_paper_b1024.log;aigis2_check8_paper_b1024_r1.log;aigis2_check8_paper_b1024_r2.log |
| aigis2 | paper | 16384 | wave64_ctrl | 1.0752 | 50137 | aigis2_wave64_ctrl_paper_b16384.log;aigis2_wave64_ctrl_paper_b16384_r1.log;aigis2_wave64_ctrl_paper_b16384_r2.log |
| aigis2 | paper | 32768 | adaptive | 1.0998 | 49197 | aigis2_adaptive_paper_b32768.log;aigis2_adaptive_paper_b32768_r1.log;aigis2_adaptive_paper_b32768_r2.log |
| aigis2 | paper | 8192 | cp_fuse | 1.2171 | 38465 | aigis2_cp_fuse_paper_b8192.log;aigis2_cp_fuse_paper_b8192_r1.log;aigis2_cp_fuse_paper_b8192_r2.log |
| aigis3 | independent | 1024 | adaptive | 1.1214 | 12032 | aigis3_adaptive_independent_b1024_r1.log;aigis3_adaptive_independent_b1024_r2.log |
| aigis3 | independent | 16384 | wave64_ctrl | 1.4200 | 42911 | aigis3_wave64_ctrl_independent_b16384_r1.log;aigis3_wave64_ctrl_independent_b16384_r2.log |
| aigis3 | independent | 32768 | cp_fuse | 1.1429 | 39925 | aigis3_cp_fuse_independent_b32768_r1.log;aigis3_cp_fuse_independent_b32768_r2.log |
| aigis3 | independent | 8192 | base | 1.0000 | 36537 | aigis3_base_independent_b8192_r1.log;aigis3_base_independent_b8192_r2.log |
| aigis3 | paper | 1024 | base | 1.0000 | 12154 | aigis3_base_paper_b1024_r1.log;aigis3_base_paper_b1024_r2.log |
| aigis3 | paper | 16384 | adaptive | 1.0331 | 41904 | aigis3_adaptive_paper_b16384_r1.log;aigis3_adaptive_paper_b16384_r2.log |
| aigis3 | paper | 32768 | base | 1.0000 | 42314 | aigis3_base_paper_b32768_r1.log;aigis3_base_paper_b32768_r2.log |
| aigis3 | paper | 8192 | adaptive | 1.2261 | 37768 | aigis3_adaptive_paper_b8192_r1.log;aigis3_adaptive_paper_b8192_r2.log |
| mldsa44 | independent | 1024 | cp_fuse | 1.3923 | 39625 | mldsa44_cp_fuse_independent_b1024.log;mldsa44_cp_fuse_independent_b1024_r1.log;mldsa44_cp_fuse_independent_b1024_r2.log |
| mldsa44 | independent | 16384 | wave64_ctrl | 1.0956 | 98699 | mldsa44_wave64_ctrl_independent_b16384.log;mldsa44_wave64_ctrl_independent_b16384_r1.log;mldsa44_wave64_ctrl_independent_b16384_r2.log |
| mldsa44 | independent | 32768 | tail16_base | 1.1110 | 100649 | mldsa44_tail16_base_independent_b32768.log;mldsa44_tail16_base_independent_b32768_r1.log;mldsa44_tail16_base_independent_b32768_r2.log |
| mldsa44 | independent | 8192 | wave64_ctrl | 1.0026 | 99262 | mldsa44_wave64_ctrl_independent_b8192.log;mldsa44_wave64_ctrl_independent_b8192_r1.log;mldsa44_wave64_ctrl_independent_b8192_r2.log |
| mldsa44 | paper | 1024 | cp_fuse | 1.3053 | 39605 | mldsa44_cp_fuse_paper_b1024.log;mldsa44_cp_fuse_paper_b1024_r1.log;mldsa44_cp_fuse_paper_b1024_r2.log |
| mldsa44 | paper | 16384 | tail16_cp_fuse | 1.0543 | 102919 | mldsa44_tail16_cp_fuse_paper_b16384.log;mldsa44_tail16_cp_fuse_paper_b16384_r1.log;mldsa44_tail16_cp_fuse_paper_b16384_r2.log |
| mldsa44 | paper | 32768 | tail16_base | 1.1936 | 98810 | mldsa44_tail16_base_paper_b32768.log;mldsa44_tail16_base_paper_b32768_r1.log;mldsa44_tail16_base_paper_b32768_r2.log |
| mldsa44 | paper | 8192 | base | 1.0000 | 96740 | mldsa44_base_paper_b8192.log;mldsa44_base_paper_b8192_r1.log;mldsa44_base_paper_b8192_r2.log |
| mldsa65 | independent | 1024 | check8 | 1.0973 | 24036 | mldsa65_check8_independent_b1024_r1.log;mldsa65_check8_independent_b1024_r2.log |
| mldsa65 | independent | 16384 | cp_fuse | 1.3625 | 62591 | mldsa65_cp_fuse_independent_b16384_r1.log;mldsa65_cp_fuse_independent_b16384_r2.log |
| mldsa65 | independent | 32768 | wave64_ctrl | 1.2118 | 60904 | mldsa65_wave64_ctrl_independent_b32768_r1.log;mldsa65_wave64_ctrl_independent_b32768_r2.log |
| mldsa65 | independent | 8192 | check16 | 1.2116 | 55130 | mldsa65_check16_independent_b8192_r1.log;mldsa65_check16_independent_b8192_r2.log |
| mldsa65 | paper | 1024 | yhat_dup | 1.0289 | 19158 | mldsa65_yhat_dup_paper_b1024_r1.log;mldsa65_yhat_dup_paper_b1024_r2.log |
| mldsa65 | paper | 16384 | tail16_base | 1.1451 | 61008 | mldsa65_tail16_base_paper_b16384_r1.log;mldsa65_tail16_base_paper_b16384_r2.log |
| mldsa65 | paper | 32768 | tail16_base | 1.1380 | 56954 | mldsa65_tail16_base_paper_b32768_r1.log;mldsa65_tail16_base_paper_b32768_r2.log |
| mldsa65 | paper | 8192 | base | 1.0000 | 61043 | mldsa65_base_paper_b8192_r1.log;mldsa65_base_paper_b8192_r2.log |
| mldsa87 | independent | 1024 | tail16 | 1.5159 | 22648 | mldsa87_tail16_independent_b1024.log |
| mldsa87 | independent | 16384 | base | 1.0000 | 46069 | mldsa87_base_independent_b16384.log;mldsa87_base_independent_b16384_r1.log;mldsa87_base_independent_b16384_r2.log |
| mldsa87 | independent | 32768 | cp_fuse | 1.0854 | 48998 | mldsa87_cp_fuse_independent_b32768.log;mldsa87_cp_fuse_independent_b32768_r1.log;mldsa87_cp_fuse_independent_b32768_r2.log |
| mldsa87 | independent | 8192 | cp_fuse | 1.0052 | 50385 | mldsa87_cp_fuse_independent_b8192.log;mldsa87_cp_fuse_independent_b8192_r1.log;mldsa87_cp_fuse_independent_b8192_r2.log |
| mldsa87 | paper | 1024 | yhat_dup | 1.1197 | 16756 | mldsa87_yhat_dup_paper_b1024.log;mldsa87_yhat_dup_paper_b1024_r1.log;mldsa87_yhat_dup_paper_b1024_r2.log |
| mldsa87 | paper | 16384 | tail16_cp_fuse | 1.1680 | 46352 | mldsa87_tail16_cp_fuse_paper_b16384.log;mldsa87_tail16_cp_fuse_paper_b16384_r1.log;mldsa87_tail16_cp_fuse_paper_b16384_r2.log |
| mldsa87 | paper | 32768 | tail16_cp_fuse | 1.1303 | 47494 | mldsa87_tail16_cp_fuse_paper_b32768.log;mldsa87_tail16_cp_fuse_paper_b32768_r1.log;mldsa87_tail16_cp_fuse_paper_b32768_r2.log |
| mldsa87 | paper | 8192 | base | 1.0000 | 55762 | mldsa87_base_paper_b8192.log;mldsa87_base_paper_b8192_r1.log;mldsa87_base_paper_b8192_r2.log |

Matrix interpretation: local wins are useful evidence. The `adaptive` row tests whether those wins can be captured in one target/mode/batch-aware build without promoting a globally regressing macro.

## AMD Limitation Evidence

The monolithic/cached-style signing candidates are retained as negative evidence; representative failures:

| target/variant | exit | hint |
| --- | ---: | --- |
| aigis1_mono_bs1_mono_bs1_b1 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| aigis1_mono_bs1_mono_bs1_b128 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| aigis1_mono_bs1_mono_bs1_b32 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| aigis1_mono_bs1_mono_bs1_b8 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| mldsa44_mono_bs1_mono_bs1_b1 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| mldsa44_mono_bs1_mono_bs1_b128 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| mldsa44_mono_bs1_mono_bs1_b32 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| mldsa44_mono_bs1_mono_bs1_b8 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| mldsa44_mono_bs2_mono_bs2_b1 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| mldsa44_mono_bs2_mono_bs2_b128 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| mldsa44_mono_bs2_mono_bs2_b32 | 1 |   [Sign] FAIL: cached/monolithic paths failed |
| mldsa44_mono_bs2_mono_bs2_b8 | 1 |   [Sign] FAIL: cached/monolithic paths failed |

## Next Tuning Step

Run `python3 amd_tools/select_sig_amd_variants.py`, inspect `amd_results/sig_amd_variant_plan.md`, then build selected variants. If `adaptive` is promoted, rerun smoke/debug/large-sweep to collect final evidence.
