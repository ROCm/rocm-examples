#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

mkdir -p amd_results/smoke

targets=(
  kyber512_amd
  kyber768_amd
  kyber1024_amd
  aigisenc1_amd
  aigisenc2_amd
  aigisenc3_amd
  aigisenc4_amd
)

batches=(1 8 32 128)

for exe in "${targets[@]}"; do
  if [[ ! -x "./${exe}" ]]; then
    echo "[skip] ./${exe} not found or not executable"
    continue
  fi

  for batch in "${batches[@]}"; do
    log="amd_results/smoke/${exe}_b${batch}.log"
    echo "[smoke] ${exe} batch=${batch}"
    stdbuf -oL -eL "./${exe}" --batch "${batch}" --n-ops 1 \
      2>&1 | tee "${log}"

    if grep -q "FAIL" "${log}"; then
      echo "[smoke] FAIL detected in ${log}" >&2
      exit 1
    fi
  done
done

python3 parse_kem_results.py amd_results/smoke > amd_results/kem_smoke_summary.csv
echo "[summary] amd_results/kem_smoke_summary.csv"
