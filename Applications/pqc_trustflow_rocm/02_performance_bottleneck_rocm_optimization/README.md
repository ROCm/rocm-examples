# Performance Bottleneck Localization And ROCm Optimization

This folder is prepared for the competition item:

```text
(2) Performance bottleneck localization and optimization
```

## What This Contains

- `kem_optimization/`: KEM profiling and tuning scripts for operation-specific TPB, launch bounds, device buffer reuse, and ROCm trace/resource analysis.
- `sig_optimization/`: ML-DSA/Aigis-sig profiling and feature-matrix tooling for resource-aware signing on ROCm.
- `evidence/`: small result summaries and decision notes suitable for PR description, report tables, and defense slides.

## Main Optimization Evidence

KEM:

- Kyber-768 keygen profiling shows the sample/XOF/rejection-sampling stage is the dominant bottleneck, about 70% of the pipeline time in the recorded run.
- Kyber-768 encaps improved from about 6.00M ops/s to about 7.10M ops/s after launch-bound tuning.
- Device buffer reuse improved continuous full-KEM throughput from about 1.93M to 2.05M instances/s in the recorded run.

Signature:

- The stable ROCm path uses a resource-aware decomp pipeline because monolithic/cached-style signing creates private-segment and scratch-pressure risk.
- Feature-matrix candidates include `adaptive`, `check8`, `check16`, `wave64_ctrl`, `cp_fuse`, `tail16_base`, `tail16_cp_fuse`, and `yhat_dup`.
- Local wins exist, for example ML-DSA-65 independent batch=16384 with `cp_fuse` reached 1.3625x speedup, and Aigis-sig3 independent batch=16384 with `wave64_ctrl` reached 1.4200x.
- Conservative selected builds keep the base decomp pipeline when no candidate satisfies the no-regression rule across all measured cells.

## Reproduction Entry Points

```bash
cd kem_optimization
bash run_kem_tune_amd.sh kyber768
bash run_kem_confirm_amd.sh kyber768
bash run_kem_final_report_amd.sh
bash run_kem_resource_profile_amd.sh kyber768 32768 200
```

```bash
cd sig_optimization
bash amd_tools/run_sig_policy_smoke.sh 128
bash amd_tools/run_sig_amd_feature_matrix.sh
python3 amd_tools/write_optimization_claims.py
bash amd_tools/run_sig_large_sweep.sh
```

## Why It Fits The Scoring Item

This folder demonstrates a complete optimization loop: workload profiling, bottleneck attribution, candidate implementation, repeated measurement, conservative promotion decisions, quantified speedups, and stability/maintainability discussion.
