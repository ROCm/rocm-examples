# AMD ROCm runbook for Kyber/Aigis-enc

This directory is the AMD/JupyterLab entry point for the KEM part of the
project. The scripts mirror the signature-side AMD workflow and keep all
generated logs under `amd_results/`.

## 1. Build

```bash
bash build_hip.sh
```

Useful overrides:

```bash
ROCM_ARCH=gfx1100 KEM_SERIAL_TPB=64 bash build_hip.sh
bash build_hip.sh kyber768
```

Outputs:

```text
kyber512_amd kyber768_amd kyber1024_amd
aigisenc1_amd aigisenc2_amd aigisenc3_amd aigisenc4_amd
amd_results/build/*.log
```

## 2. Correctness smoke test

```bash
bash run_kem_smoke_amd.sh
```

Outputs:

```text
amd_results/smoke/*.log
amd_results/kem_smoke_summary.csv
```

## 3. Batch and stream sweep

```bash
bash run_kem_sweep_amd.sh
```

Outputs:

```text
amd_results/sweep/*.log
amd_results/kem_sweep_summary.csv
amd_results/kem_best.csv
```

## 4. Profile one target

```bash
bash profile_kem_one_amd.sh kyber768_amd 8192 3
```

This runs the built-in pipeline stage timer first. If `rocprofv3` is available,
it also records a ROCm trace under `amd_results/profile/`.

## 5. Final report run

Use this after correctness and tuning are stable. It builds the stable AMD
configuration, runs final KEM throughput tests, and writes a timestamped report
directory.

```bash
bash run_kem_final_report_amd.sh
```

Expected outputs:

```text
amd_results/final_report_<timestamp>/
amd_results/final_report_<timestamp>/kem_final_extract.txt
```

The 2026-06-12 reference report used:

```text
KEM_KEYGEN_TPB=256
KEM_ENCAPS_TPB=128
KEM_DECAPS_TPB=128
Kyber batch=32768
Aigis-enc batch=65536
n_ops=20
```

## 6. ROCm resource/profile run

Use this when the paper/PPT needs bottleneck and resource evidence, not just
throughput numbers.

```bash
bash run_kem_resource_profile_amd.sh kyber768 32768 200
```

Expected outputs:

```text
amd_results/resource_profile_kyber768_<timestamp>/
amd_results/resource_profile_kyber768_<timestamp>/rocprofv3/
amd_results/resource_profile_kyber768_<timestamp>/rocm_smi_during_kyber768.log
```

Interpretation from the 2026-06-12 run:

- `rocprofv3` shows sampling/XOF/rejection sampling dominates Kyber-768
  pipeline keygen.
- Serial KEM kernels show high VGPR and scratch usage.
- `rocm-smi` shows 99%-100% GPU utilization, low VRAM pressure, and about
  237-243 W during the long Kyber-768 run.

## 7. Buffer reuse benchmark

This benchmark is useful for the final workflow/Demo because repeated file
processing should reuse device buffers instead of reallocating every round.

```bash
KEM_KEYGEN_TPB=256 KEM_ENCAPS_TPB=128 KEM_DECAPS_TPB=128 bash build_hip.sh kyber768
./kyber768_amd --batch 32768 --n-ops 5 --reuse-bench 20 --no-correctness
```

2026-06-12 reference result:

```text
Alloc-each-round: 1.93M full-KEM instances/sec
Reuse buffers:    2.05M full-KEM instances/sec
Reuse speedup:    1.064x
```

## 8. KEM tuning matrix

When continuing optimization, start with Kyber-768 because it is the clearest
mainline comparison point. The tuning script recompiles several ROCm launch and
compiler configurations, runs throughput, and emits a CSV summary.

```bash
bash run_kem_tune_amd.sh kyber768
```

For a faster first pass:

```bash
BATCH=32768 N_OPS=10 DO_CORRECTNESS=0 bash run_kem_tune_amd.sh kyber768
```

Rank results:

```bash
latest=$(ls -td amd_results/tune_kyber768_* | head -1)
sort -t, -k13,13nr "$latest/tune_summary.csv" | head
sort -t, -k14,14nr "$latest/tune_summary.csv" | head
sort -t, -k15,15nr "$latest/tune_summary.csv" | head
cat "$latest/pipeline_candidates.log"
```

Promote only stable improvements into `run_kem_final_report_amd.sh`.

## 9. All-parameter bounds probe

Use this to test all launch-bounds combinations for all seven KEM targets:

```bash
bash run_kem_all_bounds_probe_amd.sh
```

It covers:

```text
Kyber-512 / Kyber-768 / Kyber-1024
Aigis-enc-1 / Aigis-enc-2 / Aigis-enc-3 / Aigis-enc-4
bounds 000 / 001 / 010 / 011 / 100 / 101 / 110 / 111
```

Outputs:

```text
amd_results/all_bounds_probe_<timestamp>/all_bounds_probe_raw.csv
amd_results/all_bounds_probe_<timestamp>/all_bounds_probe_avg.csv
amd_results/all_bounds_probe_<timestamp>/all_bounds_probe_best.csv
```

Fast first pass:

```bash
N_OPS=10 REPEATS=1 DO_CORRECTNESS=0 bash run_kem_all_bounds_probe_amd.sh
```

Paper-grade pass:

```bash
N_OPS=30 REPEATS=2 DO_CORRECTNESS=1 bash run_kem_all_bounds_probe_amd.sh
```

## 10. All-parameter profile comparison

After the best bounds are selected, run baseline-vs-tuned ROCm profile
comparison for all seven KEM targets:

```bash
bash run_kem_all_profile_compare_amd.sh
```

It compares:

```text
baseline bounds=100
tuned bounds from all_bounds_probe_best.csv:
Kyber-512=001, Kyber-768=010, Kyber-1024=110,
Aigis-enc-1=101, Aigis-enc-2=110, Aigis-enc-3=101, Aigis-enc-4=101
```

Outputs:

```text
amd_results/profile_compare_<timestamp>/profile_compare_runs.csv
amd_results/profile_compare_<timestamp>/kernel_summary.csv
amd_results/profile_compare_<timestamp>/hip_api_summary.csv
amd_results/profile_compare_<timestamp>/key_kernel_summary.csv
amd_results/profile_compare_<timestamp>/key_kernel_compare.csv
```

Fast first pass:

```bash
N_OPS=10 PROFILE_N_OPS=1 DO_CORRECTNESS=0 bash run_kem_all_profile_compare_amd.sh
```

Paper-grade pass:

```bash
N_OPS=30 PROFILE_N_OPS=1 DO_CORRECTNESS=1 bash run_kem_all_profile_compare_amd.sh
```

## 11. ROCm toolbox pass

Use this after trace/profile comparison to probe additional ROCm tooling:

```bash
bash run_rocm_toolbox_kem_amd.sh
```

Default targets:

```text
kyber768 kyber1024 aigisenc4
```

All seven targets:

```bash
TARGETS="kyber512 kyber768 kyber1024 aigisenc1 aigisenc2 aigisenc3 aigisenc4" \
N_OPS=20 PROFILE_N_OPS=1 bash run_rocm_toolbox_kem_amd.sh
```

Outputs:

```text
amd_results/rocm_toolbox_<timestamp>/tool_discovery.txt
amd_results/rocm_toolbox_<timestamp>/rocprofv3_list_avail.txt
amd_results/rocm_toolbox_<timestamp>/toolbox_runs.csv
amd_results/rocm_toolbox_<timestamp>/*/sys_trace/
amd_results/rocm_toolbox_<timestamp>/*/pmc/
amd_results/rocm_toolbox_<timestamp>/pmc_summary.csv
```

This script attempts `rocprofv3 --sys-trace`, `rocprofv3 --pmc`, `rocm-smi`,
`rocminfo`, and `hipconfig`. Unsupported counters are skipped automatically.

## 12. Package before leaving JupyterLab

Before shutting down the AMD server, package the whole working directory from
`/app` so the raw CSV/log/profile evidence is not lost.

```bash
cd /app
tar -czf kyberandaigis-enc_amd_results_$(date +%Y%m%d_%H%M%S).tar.gz kyberandaigis-enc
ls -lh kyberandaigis-enc_amd_results_*.tar.gz
```

Download the newest `.tar.gz` from JupyterLab.

## Notes

- `build_hip.sh` uses `hipcc`, `-DUSE_HIP=1`, and `--offload-arch=gfx1100` by
  default to match the current AMD server style.
- Simple HIP migration is only a functional baseline. Use the smoke, sweep, and
  profile outputs to decide which ROCm-specific tuning path to take next.
