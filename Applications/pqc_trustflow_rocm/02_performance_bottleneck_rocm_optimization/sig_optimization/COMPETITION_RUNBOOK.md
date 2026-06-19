# AMD ROCm Competition Runbook

This is the AMD/JupyterLab source tree for the ROCm signature workload.
The local NVIDIA tree is only the 4090D/CUDA baseline and should not be used as
the AMD source of truth.

## Goal

Build a stable ROCm resource-aware signing implementation for:

- ML-DSA-44 / 65 / 87
- Aigis-sig-1 / 2 / 3

The AMD stable policy is:

```text
decomp-pipeline=on
monolithic-precomp=off
cached-precomp=off
warp-path=off
large-strategy=off
decomp-cp-fuse=off
decomp-tail=off
yhat-copy-fuse=off
decomp-adaptive=off
```

Rationale: monolithic signing can trigger ROCm private segment / scratch
resource pressure. The stable competition path uses the decomp pipeline.
The fused `cp_fuse` and tail-finish paths are implemented as AMD-specific
candidates, but the current evidence is mixed, so they are measured before
being promoted into any final target build. The new `adaptive` candidate uses
the feature matrix as a runtime policy table, so one binary can select the best
measured local policy by target, benchmark mode, and batch size, with a base
fallback on cells where aggressive AMD knobs regress.

## Build

```bash
cd /app/amd_sig_anchor_results_20260605_031411
bash amd_tools/build_sig_amd.sh
```

Expected binaries:

```bash
ls -lh mldsa44_amd mldsa65_amd mldsa87_amd aigis1_amd aigis2_amd aigis3_amd
```

## Policy Smoke

```bash
bash amd_tools/run_sig_policy_smoke.sh 128
cat amd_results/policy_smoke/policy_smoke_b128.txt
```

Required evidence:

```text
ROCm sign policy: resource-aware hybrid candidates
monolithic-precomp=off
cached-precomp=off
decomp-cp-fuse=off
decomp-tail=off
yhat-copy-fuse=off
[Sign] correctness: all 128 PASS [decomp-pipeline]
```

An adaptive selected build may instead print:

```text
[Sign] correctness: all 128 PASS [decomp-adaptive]
```

## Debug Matrix

```bash
bash amd_tools/run_sig_debug_matrix.sh
```

This checks all six targets at small batches before long sweeps.

## Large Sweep

```bash
bash amd_tools/run_sig_large_sweep.sh
```

Outputs:

```text
amd_results/large_sweep/
amd_results/sig_large_sweep_summary.csv
amd_results/sig_large_best.csv
```

Use `sig_large_best.csv` for paper/PPT throughput tables.

## AMD Feature Matrix

Run this before deciding whether the aggressive candidates should enter the
final build:

```bash
bash amd_tools/run_sig_amd_feature_matrix.sh
```

Default comparison:

```text
base              stable resource-aware decomp path
adaptive          runtime target/mode/batch policy with base fallback
check8/check16    fewer host-side done-count checks in the rejection loop
wave64_ctrl       64-thread hash/check control kernels for AMD wave64 testing
cp_fuse           fused cp*s1/cp*s2/cp*t0 pointwise products
tail16_base       small-tail finish candidate without cp_fuse
tail16_cp_fuse    small-tail finish candidate with cp_fuse
yhat_dup          cp_fuse plus sample-time y/y_hat copy candidate
```

Outputs:

```text
amd_results/sig_amd_feature_matrix.csv
amd_results/sig_amd_feature_matrix_ranked.csv
```

After the matrix:

```bash
python3 amd_tools/write_optimization_claims.py
cat amd_results/optimization_claims.md
```

Do not promote a candidate from one local win. Use the selector below; it only
recommends a non-base variant when that variant passes every measured cell for a
target and avoids measured regressions. The `adaptive` candidate is the main
way to turn mixed local wins into a single competition build without forcing one
global macro onto every workload.

For a conservative target-specific recommendation:

```bash
python3 amd_tools/select_sig_amd_variants.py
cat amd_results/sig_amd_variant_plan.md
bash amd_tools/build_sig_amd_selected.sh amd_results/sig_amd_variant_plan.env
```

Only use the selected build for final sweeps after policy smoke and debug matrix
pass again.

## Profiling

```bash
bash amd_tools/profile_sig_one.sh mldsa44_amd 1024
bash amd_tools/profile_sig_one.sh mldsa87_amd 1024
bash amd_tools/profile_sig_one.sh aigis2_amd 1024
```

Then summarize rocprof CSV output if present:

```bash
python3 amd_tools/summarize_rocm_kernel_profile.py amd_results/profile \
  > amd_results/profile/kernel_summary.csv
```

If no kernel CSV is found:

```bash
find amd_results/profile -maxdepth 4 -type f -print
sed -n '1,120p' amd_results/profile/mldsa44_amd_b1024_rocprof.log
rocprofv3 --help | head -120
```

The script no longer uses `rocprofv3 --timestamp on`, because this ROCm
environment rejected that option.

## Submission Audit

```bash
python3 amd_tools/check_competition_evidence.py
```

Expected result:

```text
[OK] sig_large_best.csv ...
[OK] policy smoke logs pass ...
[OK] large sweep logs clean ...
[OK] competition evidence audit complete
```

## Paper / PPT Story

1. Post-quantum signatures are throughput-heavy GPU workloads.
2. CUDA monolithic signing does not transfer directly to AMD ROCm.
3. AMD ROCm exposes private segment / scratch pressure for monolithic signing.
4. The project adopts a resource-aware decomp pipeline.
5. AMD-specific candidates include fused `cp*secret` pointwise work and
   wave64 control kernels; the matrix shows these are workload-sensitive.
6. The adaptive policy is the innovation layer: it applies local winners where
   they help and keeps the resource-aware baseline elsewhere.
7. Feature-matrix scripts separate proven gains from regressions and risky
   candidates.
8. Correctness and large-batch performance are reproduced by scripts.
9. ROCm profiling is used to explain kernel-level bottlenecks.
