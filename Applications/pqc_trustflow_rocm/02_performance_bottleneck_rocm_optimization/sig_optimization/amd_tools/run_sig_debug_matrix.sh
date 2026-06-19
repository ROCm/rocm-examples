#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

mkdir -p amd_results/debug

targets=(
  mldsa44_amd
  mldsa65_amd
  mldsa87_amd
  aigis1_amd
  aigis2_amd
  aigis3_amd
)

# Small batches catch correctness and resource issues quickly before a long sweep.
batches=(1 8 32 128)

for exe in "${targets[@]}"; do
  if [[ ! -x "./${exe}" ]]; then
    echo "[skip] ./${exe} not found or not executable"
    continue
  fi

  for batch in "${batches[@]}"; do
    log="amd_results/debug/${exe}_b${batch}.log"
    echo "[debug] ${exe} batch=${batch}"
    stdbuf -oL -eL "./${exe}" --batch "${batch}" --quiet --skip-keygen-oracle \
      2>&1 | tee "${log}"

    if grep -q "FAIL" "${log}"; then
      echo "[debug] FAIL detected in ${log}" >&2
      exit 1
    fi
  done
done

python3 amd_tools/parse_sig_results.py amd_results/debug > amd_results/sig_debug_summary.csv
echo "[summary] amd_results/sig_debug_summary.csv"
