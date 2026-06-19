# AMD SIG Debug Tools

These scripts are for the already-hipified ML-DSA / Aigis-sig source tree on the AMD ROCm server.

For the competition workflow, start from:

```text
COMPETITION_RUNBOOK.md
```

## Build

```bash
bash amd_tools/build_sig_amd.sh
```

The AMD build uses the ROCm resource-aware signing policy:

- `decomp-pipeline=on`
- `monolithic-precomp=off`
- `cached-precomp=off`
- `decomp-cp-fuse=off`
- `decomp-tail=off`
- `yhat-copy-fuse=off`
- `decomp-adaptive=off` in the plain stable build

The fused and tail paths are implemented, but they are kept out of the default
build until the AMD feature matrix proves a target-specific win. Current
evidence shows `cp_fuse` can help some ML-DSA cases while regressing other
targets and batch sizes, so the stable competition build stays with the base
resource-aware decomp pipeline. The `adaptive` matrix candidate is different:
it keeps one binary but selects measured local winners at runtime by
target/mode/batch, and falls back to base where the matrix shows a regression.

This builds:

- `mldsa44_amd`
- `mldsa65_amd`
- `mldsa87_amd`
- `aigis1_amd`
- `aigis2_amd`
- `aigis3_amd`

## Sweep

```bash
bash amd_tools/run_sig_sweep.sh
```

Logs are written to `amd_results/sweep/`, and the CSV summary is written to:

```text
amd_results/sig_sweep_summary.csv
```

## Large Batch Sweep

Use this before kernel-level tuning to find the AMD batch-size ceiling and the best Keygen/Sign/Verify throughput:

```bash
bash amd_tools/run_sig_large_sweep.sh
```

It runs all six signature targets with both `--bench-paper` and `--bench-independent` for batch sizes `8192, 16384, 32768`.

Outputs:

```text
amd_results/large_sweep/
amd_results/sig_large_sweep_summary.csv
amd_results/sig_large_best.csv
```

To compare against a 4090 CSV after uploading it to the server:

```bash
python3 amd_tools/compare_amd_4090.py amd_results/sig_large_sweep_summary.csv /app/4090数据.csv > amd_results/amd_vs_4090_large.csv
```

## AMD Feature Matrix

Use this after a correctness smoke test to compare the stable build against
candidate AMD-specific signing variants:

```bash
bash amd_tools/run_sig_amd_feature_matrix.sh
```

Default matrix:

```text
targets:  mldsa44 mldsa87 aigis2
batches:  1024 8192
modes:    independent paper
variants: base adaptive check8 check16 wave64_ctrl cp_fuse tail16_base tail16_cp_fuse yhat_dup
```

Candidate meanings:

```text
base              stable resource-aware decomp path
adaptive          runtime target/mode/batch policy with base fallback
check8/check16    fewer host-side done-count checks in the rejection loop
wave64_ctrl       64-thread hash/check control kernels for AMD wave64 testing
cp_fuse           fused cp*s1/cp*s2/cp*t0 pointwise products
tail16_base       small-tail finish without cp_fuse
tail16_cp_fuse    small-tail finish with cp_fuse
yhat_dup          sample-time y/y_hat copy candidate
```

Outputs:

```text
amd_results/sig_amd_feature_matrix.csv
amd_results/sig_amd_feature_matrix_ranked.csv
```

For a wider pass:

```bash
FEATURE_TARGETS="mldsa44 mldsa65 mldsa87 aigis1 aigis2 aigis3" \
FEATURE_BATCHES="1024 8192 16384" \
bash amd_tools/run_sig_amd_feature_matrix.sh
```

Generate a report-ready summary:

```bash
python3 amd_tools/write_optimization_claims.py
```

After the matrix, generate a conservative per-target build plan:

```bash
python3 amd_tools/select_sig_amd_variants.py
cat amd_results/sig_amd_variant_plan.md
```

If `adaptive` appears as a local winner or selected variant, rerun policy smoke
and debug matrix before using it for final large sweeps.

## Debug Matrix

After every source change, run a quick correctness/resource smoke test first:

```bash
bash amd_tools/run_sig_debug_matrix.sh
```

This runs all six signature targets with batch sizes `1, 8, 32, 128` and writes:

```text
amd_results/debug/
amd_results/sig_debug_summary.csv
```

## Profile One Target

```bash
bash amd_tools/profile_sig_one.sh mldsa44_amd 1024
```

The script runs the executable with `--profile`. If `rocprofv3` is installed, it also records a ROCm profile under `amd_results/profile/`.

Summarize any ROCm kernel CSV data:

```bash
python3 amd_tools/summarize_rocm_kernel_profile.py amd_results/profile > amd_results/profile/kernel_summary.csv
```

## Submission Evidence Audit

```bash
python3 amd_tools/check_competition_evidence.py
```
