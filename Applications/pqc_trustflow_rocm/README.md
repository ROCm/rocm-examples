# PR Upload Ready Package

This directory is split by the two innovation-development scoring items in the AMD competition.

## 01_unsupported_feature_rocm_pqc_api

Corresponds to:

```text
(1) Development of currently unsupported functions
```

This folder contains the ROCm/HIP post-quantum cryptography function layer:

- batch KEM implementation and file-level KEM CLI API;
- batch ML-DSA/Aigis-sig implementation and file-level signature CLI API;
- TrustFlow frontend integration that calls the ROCm backends for multi-file secure packaging;
- quick-start and API notes.

Use this folder when the PR needs to highlight new ROCm backend functionality and upper-layer API adaptation.

## 02_performance_bottleneck_rocm_optimization

Corresponds to:

```text
(2) Performance bottleneck localization and optimization
```

This folder contains the profiling, tuning, and evidence layer:

- KEM tuning scripts for TPB, launch bounds, buffer reuse, and ROCm profiling;
- signature tuning scripts for resource-aware decomp pipeline and feature-matrix candidates;
- small evidence summaries for KEM final throughput, SIG large sweep, local winners, and optimization decisions;
- original analysis notes showing bottleneck attribution and conservative promotion decisions.

Use this folder when the PR needs to highlight systematic performance analysis, quantified optimization, and engineering trade-offs.

## Excluded From PR

Generated binaries, caches, secret files, large logs, temporary outputs, and competition-only documents are intentionally excluded.
