#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

mkdir -p amd_results/sweep

targets=(
  mldsa44_amd
  mldsa65_amd
  mldsa87_amd
  aigis1_amd
  aigis2_amd
  aigis3_amd
)

batches=(128 512 1024 2048 4096)

for exe in "${targets[@]}"; do
  if [[ ! -x "./${exe}" ]]; then
    echo "[skip] ./${exe} not found or not executable"
    continue
  fi

  for batch in "${batches[@]}"; do
    log="amd_results/sweep/${exe}_b${batch}.log"
    echo "[run] ${exe} batch=${batch}"
    stdbuf -oL -eL "./${exe}" --batch "${batch}" --quiet --skip-keygen-oracle \
      2>&1 | tee "${log}"
  done
done

python3 amd_tools/parse_sig_results.py amd_results/sweep > amd_results/sig_sweep_summary.csv
echo "[summary] amd_results/sig_sweep_summary.csv"
