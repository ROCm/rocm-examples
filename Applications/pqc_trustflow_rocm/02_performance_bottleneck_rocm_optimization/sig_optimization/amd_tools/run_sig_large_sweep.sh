#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

mkdir -p amd_results/large_sweep

targets=(
  mldsa44_amd
  mldsa65_amd
  mldsa87_amd
  aigis1_amd
  aigis2_amd
  aigis3_amd
)

modes=(
  paper
  independent
)

# First boundary pass. Add 65536 only after these are stable.
batches=(8192 16384 32768)

for exe in "${targets[@]}"; do
  if [[ ! -x "./${exe}" ]]; then
    echo "[skip] ./${exe} not found or not executable"
    continue
  fi

  for mode in "${modes[@]}"; do
    mode_flag="--bench-paper"
    if [[ "${mode}" == "independent" ]]; then
      mode_flag="--bench-independent"
    fi

    for batch in "${batches[@]}"; do
      log="amd_results/large_sweep/${exe}_${mode}_b${batch}.log"
      echo "[large] ${exe} mode=${mode} batch=${batch}"
      set +e
      stdbuf -oL -eL "./${exe}" "${mode_flag}" --batch "${batch}" --quiet --skip-keygen-oracle \
        2>&1 | tee "${log}"
      rc=${PIPESTATUS[0]}
      set -e
      echo "[large] exit_code=${rc}" | tee -a "${log}"
    done
  done
done

python3 amd_tools/parse_sig_results.py amd_results/large_sweep > amd_results/sig_large_sweep_summary.csv
python3 amd_tools/summarize_sig_best.py amd_results/sig_large_sweep_summary.csv > amd_results/sig_large_best.csv

echo "[summary] amd_results/sig_large_sweep_summary.csv"
echo "[summary] amd_results/sig_large_best.csv"
