# KEM AMD Optimization Log

This log records reproducible optimization steps for the Kyber/Aigis-enc KEM
module on AMD ROCm. Use it as the source material for paper/PPT tables.

## Current Stable Configuration

```text
KEM_KEYGEN_TPB=256
KEM_ENCAPS_TPB=128
KEM_DECAPS_TPB=128
Kyber batch=32768
Aigis-enc batch=65536
n_ops=20 for final throughput tables
```

## Verified Optimizations

1. Operation-specific TPB tuning
   - Replaced one shared `KEM_SERIAL_TPB` path with `KEM_KEYGEN_TPB`,
     `KEM_ENCAPS_TPB`, and `KEM_DECAPS_TPB`.
   - Kyber-768 keygen benefits from larger keygen TPB on gfx1100.

2. Device buffer reuse
   - Added `--reuse-bench <rounds>`.
   - Kyber-768 full-KEM continuous batch throughput improved from
     1.93M to 2.05M instances/s in the measured run, a 1.064x speedup.

3. ROCm trace-based bottleneck localization
   - `rocprofv3` kernel trace shows `batch_keygen_warp_sample_kernel`
     dominates the pipeline keygen path.
   - Monolithic serial KEM kernels show high VGPR and scratch usage.
   - NTT/matvec kernels are not the primary bottleneck in the current run.

## Reproducible Scripts

Final throughput table:

```bash
bash run_kem_final_report_amd.sh
```

Resource and ROCm trace profile:

```bash
bash run_kem_resource_profile_amd.sh kyber768 32768 200
```

Buffer reuse benchmark:

```bash
KEM_KEYGEN_TPB=256 KEM_ENCAPS_TPB=128 KEM_DECAPS_TPB=128 bash build_hip.sh kyber768
./kyber768_amd --batch 32768 --n-ops 5 --reuse-bench 20 --no-correctness
```

## Paper-Ready Findings

- Simple HIP migration is functional but not enough for peak performance.
- ROCm/RDNA3-specific TPB tuning improves Kyber keygen throughput.
- Device buffer reuse improves end-to-end full-KEM workflow throughput.
- `rocprofv3` confirms sample/XOF/rejection sampling is the priority kernel
  optimization target, while `rocm-smi` shows high GPU utilization and low VRAM
  pressure during long Kyber-768 runs.

## 2026-06-12 Next Optimization Pass

Added a ROCm tuning matrix for the KEM module.

Code changes:

- `build_hip.sh` now accepts `OPT_LEVEL`, `KEM_KEYPAIR_LAUNCH_BOUNDS`,
  `KEM_ENCAPS_LAUNCH_BOUNDS`, `KEM_DECAPS_LAUNCH_BOUNDS`,
  `WP_KG_WARPS_BLOCK`, `KEM_PACK_TPB`, and `EXTRA_HIPCC_FLAGS`.
- `batch_kem.cuh` now allows `WP_KG_WARPS_BLOCK` and `KEM_PACK_TPB` to be
  provided at compile time.
- `run_kem_tune_amd.sh` sweeps the current serial final-report path first, then
  checks pipeline sampling/pack candidates.

Run on AMD:

```bash
bash run_kem_tune_amd.sh kyber768
```

Useful quicker runs:

```bash
BATCH=32768 N_OPS=10 DO_CORRECTNESS=0 bash run_kem_tune_amd.sh kyber768
BATCH=65536 N_OPS=10 DO_CORRECTNESS=0 bash run_kem_tune_amd.sh aigisenc4
```

After the run, rank candidates:

```bash
latest=$(ls -td amd_results/tune_kyber768_* | head -1)
sort -t, -k13,13nr "$latest/tune_summary.csv" | head
sort -t, -k14,14nr "$latest/tune_summary.csv" | head
sort -t, -k15,15nr "$latest/tune_summary.csv" | head
cat "$latest/pipeline_candidates.log"
```

Decision rule:

- If a candidate improves one operation without hurting the other two, promote
  it to `run_kem_final_report_amd.sh`.
- If the best keygen, encaps, and decaps candidates differ, keep
  operation-specific TPB values instead of forcing one shared setting.
- If pipeline `sample` improves but total keygen does not, keep it as a
  profiling result rather than final default.

### Kyber-768 first tuning result

First AMD run:

```text
amd_results/tune_kyber768_20260612_085500/tune_summary.csv
```

Main findings from the pasted result:

| Candidate | Keygen | Encaps | Decaps | Interpretation |
|---|---:|---:|---:|---|
| O2 kg=256 enc=128 dec=128 bounds=1/0/0 | 6.28M | 6.00M | 5.64M | Current stable neighborhood |
| O2 kg=256 enc=128 dec=128 bounds=0/1/0 | 6.27M | 7.12M | 5.65M | Best balanced candidate |
| O2 kg=256 enc=128 dec=128 bounds=1/1/0 | 6.22M | 7.14M | 5.64M | Highest encaps candidate |
| O3 kg=256 enc=128 dec=128 bounds=1/0/0 | 6.29M | 5.98M | 5.64M | Highest keygen candidate |
| O3 kg=512 enc=128 dec=128 bounds=1/0/0 | 5.71M | 5.99M | 5.66M | Highest decaps candidate, but hurts keygen |

Conclusion:

```text
The first clear optimization signal is KEM_ENCAPS_LAUNCH_BOUNDS=1 for
Kyber-768. It improves encaps from about 6.0M ops/s to about 7.1M ops/s while
keygen and decaps remain close to the previous stable region.
```

Next confirmation run:

```bash
bash run_kem_confirm_amd.sh kyber768
```

Fast confirmation:

```bash
N_OPS=30 REPEATS=2 bash run_kem_confirm_amd.sh kyber768
```

If `balanced_encbounds_o2_b010` or `encbest_o2_b110` remains stable, promote
`KEM_ENCAPS_LAUNCH_BOUNDS=1` to the Kyber-768 final configuration.

### Kyber-768 confirmation result

Confirmation run:

```text
amd_results/confirm_kyber768_20260612_090854/confirm_summary.csv
batch=32768 n_ops=50 repeats=3
```

Average of three repeats:

| Tag | Keygen avg | Encaps avg | Decaps avg | Decision |
|---|---:|---:|---:|---|
| baseline_o2_256_128_128_b100 | 6.485M | 6.004M | 5.490M | Old stable baseline |
| balanced_encbounds_o2_b010 | 6.475M | 7.102M | 5.435M | Promote |
| encbest_o2_b110 | 6.466M | 7.101M | 5.429M | Similar encaps, slightly lower keygen/decaps |
| keygenbest_o3_b100 | 6.502M | 5.993M | 5.471M | No encaps gain |
| decbest_o3_kg512_b100 | 5.747M | 6.000M | 5.482M | Hurts keygen |

Confirmed improvement:

```text
Kyber-768 encaps improved from about 6.00M ops/s to about 7.10M ops/s
(~18.3% relative improvement) by enabling KEM_ENCAPS_LAUNCH_BOUNDS=1 while
using keypair_bounds=0 and decaps_bounds=0 for the final Kyber-768 build.
```

Trade-off:

```text
The tuned config reduces decaps by about 1.0% and keygen by about 0.2% in the
repeat average, which is acceptable because encaps gains about 18%.
```

Promoted final Kyber-768 config:

```text
OPT_LEVEL=O2
KEM_KEYGEN_TPB=256
KEM_ENCAPS_TPB=128
KEM_DECAPS_TPB=128
KEM_KEYPAIR_LAUNCH_BOUNDS=0
KEM_ENCAPS_LAUNCH_BOUNDS=1
KEM_DECAPS_LAUNCH_BOUNDS=0
```

`run_kem_final_report_amd.sh` now applies this tuned config to Kyber-768 only.

### Tuned final report result

Tuned final report pasted from AMD:

```text
Kyber-512:   Keygen 10.050M, Encaps 11.342M, Decaps 7.448M ops/s
Kyber-768:   Keygen  6.229M, Encaps  7.151M, Decaps 5.640M ops/s
Kyber-1024:  Keygen  4.470M, Encaps  4.290M, Decaps 3.829M ops/s
Aigis-enc-1: Keygen 10.299M, Encaps  8.204M, Decaps 5.769M ops/s
Aigis-enc-2: Keygen  6.671M, Encaps  5.240M, Decaps 3.841M ops/s
Aigis-enc-3: Keygen  6.284M, Encaps  5.145M, Decaps 3.309M ops/s
Aigis-enc-4: Keygen  4.147M, Encaps  3.450M, Decaps 2.564M ops/s
```

Compared with the previous final report, the key paper-worthy changes are:

| Target | Operation | Previous | Tuned | Change |
|---|---:|---:|---:|---:|
| Kyber-768 | Encaps | 5.999M | 7.151M | +19.2% |
| Aigis-enc-1 | Encaps | 7.205M | 8.204M | +13.9% |
| Aigis-enc-2 | Encaps | 4.704M | 5.240M | +11.4% |
| Aigis-enc-4 | Encaps | 2.951M | 3.450M | +16.9% |

Notes:

- Kyber-768 encaps tuning is confirmed by repeat testing and is now final.
- Aigis-enc also benefited in encaps under the new final-report configuration,
  but Aigis-enc-3 decaps regressed in this run. Do not generalize the Kyber
  decision to all Aigis variants until the Aigis-specific bounds probe is done.

### Aigis-enc-4 bounds probe result

Probe run:

```text
amd_results/bounds_probe_aigisenc4_20260612_093104/bounds_probe_summary.csv
batch=65536 n_ops=30 repeats=2
```

Average of two repeats:

| Tag | Bounds | Keygen avg | Encaps avg | Decaps avg | Decision |
|---|---|---:|---:|---:|---|
| baseline_b100 | 1/0/0 | 4.175M | 3.451M | 2.570M | Old baseline |
| encbounds_b010 | 0/1/0 | 4.159M | 3.724M | 2.573M | Good |
| encbounds_b110 | 1/1/0 | 4.173M | 3.721M | 2.571M | Promote |
| allbounds_b111 | 1/1/1 | 4.174M | 2.953M | 2.432M | Reject |

Confirmed improvement:

```text
Aigis-enc-4 encaps improved from about 3.45M ops/s to about 3.72M ops/s
(~7.9% relative improvement) by enabling KEM_ENCAPS_LAUNCH_BOUNDS=1.
```

Promoted final Aigis-enc-4 config:

```text
OPT_LEVEL=O2
KEM_KEYGEN_TPB=256
KEM_ENCAPS_TPB=128
KEM_DECAPS_TPB=128
KEM_KEYPAIR_LAUNCH_BOUNDS=1
KEM_ENCAPS_LAUNCH_BOUNDS=1
KEM_DECAPS_LAUNCH_BOUNDS=0
```

Rejected config:

```text
KEM_DECAPS_LAUNCH_BOUNDS=1 should not be enabled for Aigis-enc-4 because
`allbounds_b111` lowers encaps and decaps substantially.
```

### 2026-06-14 all-parameter bounds probe

Probe run pasted from AMD:

```text
amd_results/all_bounds_probe_<timestamp>/
N_OPS=30 REPEATS=2 DO_CORRECTNESS=1
Kyber batch=32768
Aigis-enc batch=65536
```

The script tested all 8 launch-bounds combinations for all 7 KEM targets:

```text
000 / 001 / 010 / 011 / 100 / 101 / 110 / 111
```

Best balanced configurations from `all_bounds_probe_best.csv`:

| Target | Bounds | Keygen avg | Encaps avg | Decaps avg | Balanced score |
|---|---|---:|---:|---:|---:|
| Kyber-512 | 001 | 10.432M | 11.398M | 8.508M | 10.241M |
| Kyber-768 | 010 | 6.385M | 7.167M | 5.584M | 6.458M |
| Kyber-1024 | 110 | 4.517M | 4.908M | 3.819M | 4.464M |
| Aigis-enc-1 | 101 | 10.478M | 8.217M | 6.497M | 8.379M |
| Aigis-enc-2 | 110 | 6.709M | 5.627M | 3.830M | 5.413M |
| Aigis-enc-3 | 101 | 6.385M | 5.152M | 4.057M | 5.193M |
| Aigis-enc-4 | 101 | 4.171M | 3.447M | 2.962M | 3.518M |

Interpretation:

```text
There is no single best launch-bounds setting for all KEM targets.
ROCm tuning must be parameter-set-aware.
```

Important detailed findings:

- `encaps_bounds=1` is consistently strong for Kyber-768, Kyber-1024,
  Aigis-enc-1, Aigis-enc-2, and Aigis-enc-4 encaps.
- `decaps_bounds=1` strongly improves Kyber-512, Aigis-enc-1,
  Aigis-enc-3, and Aigis-enc-4 decaps.
- Enabling all bounds `111` is often a negative optimization, especially for
  Kyber-768 and Aigis-enc-4.

The final report script now uses the best balanced per-target configuration:

```text
Kyber-512   bounds=001
Kyber-768   bounds=010
Kyber-1024  bounds=110
Aigis-enc-1 bounds=101
Aigis-enc-2 bounds=110
Aigis-enc-3 bounds=101
Aigis-enc-4 bounds=101
```

### 2026-06-14 balanced-best final report

Final report pasted from AMD:

```text
amd_results/final_report_20260614_012319/kem_final_extract.txt
```

| Target | Bounds | Keygen ops/s | Encaps ops/s | Decaps ops/s |
|---|---|---:|---:|---:|
| Kyber-512 | 001 | 10,095,164 | 11,368,410 | 8,451,971 |
| Kyber-768 | 010 | 6,276,945 | 7,142,451 | 5,651,891 |
| Kyber-1024 | 110 | 4,447,101 | 4,916,932 | 3,829,267 |
| Aigis-enc-1 | 101 | 10,240,547 | 8,204,293 | 6,497,139 |
| Aigis-enc-2 | 110 | 6,605,967 | 5,630,086 | 3,827,435 |
| Aigis-enc-3 | 101 | 6,305,602 | 5,144,497 | 4,060,120 |
| Aigis-enc-4 | 101 | 4,156,445 | 3,444,781 | 2,961,419 |

Paper-ready statement:

```text
After parameter-set-aware ROCm launch-bounds tuning, all seven KEM targets
retain PASS correctness evidence and reach million-to-ten-million-level
throughput. The tuning improves different operations for different parameter
sets, proving that ROCm PQC kernels need per-parameter resource policies rather
than a single global launch configuration.
```

### Next: all-parameter profile comparison

Added scripts:

```text
run_kem_all_profile_compare_amd.sh
summarize_profile_compare.py
```

Purpose:

```text
Compare baseline bounds=100 against tuned per-target bounds for all seven KEM
targets using rocprofv3 kernel trace and HIP API trace.
```

Run on AMD:

```bash
N_OPS=30 PROFILE_N_OPS=1 DO_CORRECTNESS=1 bash run_kem_all_profile_compare_amd.sh
```

Outputs:

```text
amd_results/profile_compare_<timestamp>/profile_compare_runs.csv
amd_results/profile_compare_<timestamp>/kernel_summary.csv
amd_results/profile_compare_<timestamp>/hip_api_summary.csv
amd_results/profile_compare_<timestamp>/key_kernel_summary.csv
amd_results/profile_compare_<timestamp>/key_kernel_compare.csv
```

The key file for paper analysis is:

```text
key_kernel_compare.csv
```

It reports keypair/encaps/decaps serial kernel time, VGPR, SGPR, scratch,
workgroup, and tuned-vs-baseline percentage change.

### 2026-06-14 all-parameter profile comparison result

Profile comparison pasted from AMD:

```text
amd_results/profile_compare_<timestamp>/
profile_compare_runs.csv
key_kernel_compare.csv
key_kernel_summary.csv
```

Throughput change from baseline bounds `100` to tuned per-target bounds:

| Target | Tuned bounds | Keygen change | Encaps change | Decaps change | Main gain |
|---|---|---:|---:|---:|---|
| Kyber-512 | 001 | -0.3% | -0.2% | +12.9% | Decaps |
| Kyber-768 | 010 | +0.4% | +19.2% | -0.2% | Encaps |
| Kyber-1024 | 110 | +1.2% | +14.2% | -0.9% | Encaps |
| Aigis-enc-1 | 101 | -0.9% | ~0.0% | +12.6% | Decaps |
| Aigis-enc-2 | 110 | -0.1% | +7.4% | -0.1% | Encaps |
| Aigis-enc-3 | 101 | -0.1% | -0.1% | +22.7% | Decaps |
| Aigis-enc-4 | 101 | ~0.0% | +0.1% | +15.2% | Decaps |

Key kernel time changes from `key_kernel_compare.csv`:

| Target | Operation improved | Kernel time change | Resource change |
|---|---|---:|---|
| Kyber-512 | Decaps | -11.84% | VGPR 184 -> 200, scratch 14784 -> 14752 |
| Kyber-768 | Encaps | -16.51% | VGPR 184 -> 200, scratch 16064 -> 16048 |
| Kyber-1024 | Encaps | -12.99% | VGPR 184 -> 200, scratch 18144 -> 18128 |
| Aigis-enc-1 | Decaps | -10.50% | VGPR 184 -> 200, scratch 14720 -> 14704 |
| Aigis-enc-2 | Encaps | -6.22% | VGPR 184 -> 200, scratch 16032 -> 16016 |
| Aigis-enc-3 | Decaps | -19.34% | VGPR 184 -> 200, scratch 17088 -> 17072 |
| Aigis-enc-4 | Decaps | -13.53% | VGPR 184 -> 200, scratch 19648 -> 19632 |

Important interpretation:

```text
The tuned launch-bounds setting does not simply reduce register count.
For the operations that improve most, VGPR often increases from 184 to 200
while scratch decreases slightly by 16-32 bytes and kernel runtime drops
significantly. This suggests the improvement comes from ROCm compiler scheduling
and occupancy/resource trade-offs, not from a naive "fewer registers is always
better" rule.
```

Per-target analysis:

- **Kyber-512**: tuned `001` is a decaps-focused configuration. Decaps kernel
  time drops 11.84%, matching the 12.9% throughput gain. Keygen/encaps are
  almost unchanged in throughput, although profile keypair kernel time is
  noisier and increases in the one-iteration trace.
- **Kyber-768**: tuned `010` is a clean encaps optimization. Encaps kernel time
  drops 16.51% and throughput rises 19.2%, while keygen/decaps remain stable.
  This is the strongest Kyber example for the paper.
- **Kyber-1024**: tuned `110` improves encaps kernel time by 12.99% and
  throughput by 14.2%. It also slightly improves keypair kernel time, while
  decaps is roughly unchanged.
- **Aigis-enc-1**: tuned `101` mainly improves decaps. Decaps kernel time drops
  10.50% and decaps throughput rises 12.6%. Encaps remains essentially equal.
- **Aigis-enc-2**: tuned `110` mainly improves encaps. Encaps kernel time drops
  6.22% and throughput rises 7.4%. Keypair profile time worsens, so this config
  should be described as operation-selective rather than globally faster.
- **Aigis-enc-3**: tuned `101` is a strong decaps optimization. Decaps kernel
  time drops 19.34% and throughput rises 22.7%, with keygen/encaps nearly
  unchanged in throughput.
- **Aigis-enc-4**: tuned `101` improves decaps by 15.2% in throughput and
  lowers decaps kernel time by 13.53%; keygen and encaps are effectively
  unchanged.

Paper-ready conclusion:

```text
ROCm launch-bounds tuning changes the compiler's resource scheduling decisions.
For PQC KEM kernels with large private state and scratch pressure, the best
configuration is operation- and parameter-set-specific. The measured wins are
not explained by global GPU utilization or VRAM capacity, but by per-kernel
resource trade-offs visible in rocprofv3: key operation runtime drops while VGPR
and scratch shift slightly.
```

### Next: ROCm toolbox pass

Added:

```text
run_rocm_toolbox_kem_amd.sh
summarize_rocm_pmc.py
```

Purpose:

```text
Probe additional ROCm tools beyond kernel/HIP trace:
tool discovery, rocprofv3 --list-avail, rocprofv3 --sys-trace,
rocprofv3 --pmc hardware counters, rocm-smi sampling, rocminfo, hipconfig,
and rocprof-compute availability.
```

Representative run:

```bash
bash run_rocm_toolbox_kem_amd.sh
```

All-target run:

```bash
TARGETS="kyber512 kyber768 kyber1024 aigisenc1 aigisenc2 aigisenc3 aigisenc4" \
N_OPS=20 PROFILE_N_OPS=1 bash run_rocm_toolbox_kem_amd.sh
```

Expected outputs:

```text
amd_results/rocm_toolbox_<timestamp>/tool_discovery.txt
amd_results/rocm_toolbox_<timestamp>/rocprofv3_list_avail.txt
amd_results/rocm_toolbox_<timestamp>/toolbox_runs.csv
amd_results/rocm_toolbox_<timestamp>/pmc_summary.csv
```

This is an exploratory pass. If the AMD JupyterLab image exposes only a subset
of ROCm counters or lacks `rocprof-compute`, the script records that instead of
failing the workflow.
