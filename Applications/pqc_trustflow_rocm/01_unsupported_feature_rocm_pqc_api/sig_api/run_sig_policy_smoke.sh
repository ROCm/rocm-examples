#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

out_dir="amd_results/policy_smoke"
mkdir -p "${out_dir}"

targets=(
  mldsa44_amd
  mldsa65_amd
  mldsa87_amd
  aigis1_amd
  aigis2_amd
  aigis3_amd
)

batch="${1:-128}"
summary="${out_dir}/policy_smoke_b${batch}.txt"
: > "${summary}"

echo "[policy-smoke] batch=${batch}" | tee -a "${summary}"

for exe in "${targets[@]}"; do
  if [[ ! -x "./${exe}" ]]; then
    echo "[skip] ./${exe} not found or not executable" | tee -a "${summary}"
    continue
  fi

  log="${out_dir}/${exe}_b${batch}.log"
  echo "[run] ${exe} batch=${batch}" | tee -a "${summary}"
  stdbuf -oL -eL "./${exe}" --batch "${batch}" --quiet --skip-keygen-oracle \
    2>&1 | tee "${log}"

  grep -E "ROCm sign policy|monolithic-precomp|decomp-cp-fuse|decomp-tail|yhat-copy-fuse|decomp-adaptive|rationale|\\[Sign\\] correctness|  Sign[[:space:]]+" "${log}" \
    | sed "s/^/[${exe}] /" | tee -a "${summary}"

  if grep -q "FAIL" "${log}"; then
    echo "[policy-smoke] FAIL detected in ${log}" | tee -a "${summary}" >&2
    exit 1
  fi
done

echo "[policy-smoke] PASS; summary=${summary}"
