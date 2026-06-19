#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

N_OPS=${N_OPS:-20}
PROFILE_N_OPS=${PROFILE_N_OPS:-1}
DO_CORRECTNESS=${DO_CORRECTNESS:-0}
KYBER_BATCH=${KYBER_BATCH:-32768}
AIGIS_BATCH=${AIGIS_BATCH:-65536}
KEM_KEYGEN_TPB=${KEM_KEYGEN_TPB:-256}
KEM_ENCAPS_TPB=${KEM_ENCAPS_TPB:-128}
KEM_DECAPS_TPB=${KEM_DECAPS_TPB:-128}
OPT_LEVEL=${OPT_LEVEL:-O2}

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="amd_results/profile_compare_${stamp}"
mkdir -p "${out_dir}"

runs_csv="${out_dir}/profile_compare_runs.csv"
echo "target,config,bounds,batch,n_ops,opt,kg_tpb,enc_tpb,dec_tpb,keypair_bounds,encaps_bounds,decaps_bounds,keygen_ops_s,encaps_ops_s,decaps_ops_s,status,run_dir" > "${runs_csv}"

extract_metric() {
  local label="$1"
  local log="$2"
  grep -E "  ${label}:" "${log}" \
    | tail -1 \
    | grep -oE '[0-9]+ ops/sec' \
    | tail -1 \
    | awk '{print $1}'
}

batch_for_target() {
  local target="$1"
  if [[ "${target}" == kyber* ]]; then
    echo "${KYBER_BATCH}"
  else
    echo "${AIGIS_BATCH}"
  fi
}

tuned_bounds_for_target() {
  case "$1" in
    kyber512)  echo "001" ;;
    kyber768)  echo "010" ;;
    kyber1024) echo "110" ;;
    aigisenc1) echo "101" ;;
    aigisenc2) echo "110" ;;
    aigisenc3) echo "101" ;;
    aigisenc4) echo "101" ;;
    *) echo "100" ;;
  esac
}

run_one() {
  local target="$1"
  local config="$2"
  local bounds="$3"
  local kb="${bounds:0:1}"
  local eb="${bounds:1:1}"
  local db="${bounds:2:1}"
  local batch
  batch="$(batch_for_target "${target}")"
  local run_name="${target}_${config}_bounds${bounds}"
  local run_dir="${out_dir}/${run_name}"
  mkdir -p "${run_dir}/rocprofv3"
  local status="PASS"

  {
    echo "target=${target}"
    echo "config=${config}"
    echo "bounds=${bounds}"
    echo "batch=${batch}"
    echo "n_ops=${N_OPS}"
    echo "profile_n_ops=${PROFILE_N_OPS}"
    echo "opt=${OPT_LEVEL}"
    echo "tpb=${KEM_KEYGEN_TPB}/${KEM_ENCAPS_TPB}/${KEM_DECAPS_TPB}"
    echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
    hipcc --version || true
  } > "${run_dir}/metadata.txt" 2>&1

  echo
  echo "[run] ${target} ${config} bounds=${bounds} batch=${batch}"

  OPT_LEVEL="${OPT_LEVEL}" \
  KEM_KEYGEN_TPB="${KEM_KEYGEN_TPB}" KEM_ENCAPS_TPB="${KEM_ENCAPS_TPB}" KEM_DECAPS_TPB="${KEM_DECAPS_TPB}" \
  KEM_KEYPAIR_LAUNCH_BOUNDS="${kb}" KEM_ENCAPS_LAUNCH_BOUNDS="${eb}" KEM_DECAPS_LAUNCH_BOUNDS="${db}" \
    bash build_hip.sh "${target}" > "${run_dir}/build.log" 2>&1 || status="FAIL"

  if [[ "${status}" == "PASS" && "${DO_CORRECTNESS}" == "1" ]]; then
    "./${target}_amd" --batch 128 --n-ops 1 > "${run_dir}/correctness.log" 2>&1 || status="FAIL"
  fi

  if [[ "${status}" == "PASS" ]]; then
    "./${target}_amd" --batch "${batch}" --n-ops "${N_OPS}" --no-correctness \
      > "${run_dir}/benchmark.log" 2>&1 || status="FAIL"
  fi

  local kg_ops=""
  local enc_ops=""
  local dec_ops=""
  if [[ "${status}" == "PASS" ]]; then
    kg_ops="$(extract_metric Keygen "${run_dir}/benchmark.log" || true)"
    enc_ops="$(extract_metric Encaps "${run_dir}/benchmark.log" || true)"
    dec_ops="$(extract_metric Decaps "${run_dir}/benchmark.log" || true)"

    if command -v rocprofv3 >/dev/null 2>&1; then
      rocprofv3 \
        --kernel-trace \
        --hip-trace \
        --output-format csv \
        --output-directory "${run_dir}/rocprofv3" \
        -- \
        "./${target}_amd" --batch "${batch}" --n-ops "${PROFILE_N_OPS}" --no-correctness \
        > "${run_dir}/rocprofv3.log" 2>&1 || true
    else
      echo "rocprofv3 not found" > "${run_dir}/rocprofv3.log"
    fi
  fi

  echo "${target},${config},${bounds},${batch},${N_OPS},${OPT_LEVEL},${KEM_KEYGEN_TPB},${KEM_ENCAPS_TPB},${KEM_DECAPS_TPB},${kb},${eb},${db},${kg_ops},${enc_ops},${dec_ops},${status},${run_name}" | tee -a "${runs_csv}"
}

targets=(kyber512 kyber768 kyber1024 aigisenc1 aigisenc2 aigisenc3 aigisenc4)

echo "[profile-compare] out=${out_dir}"
echo "[profile-compare] N_OPS=${N_OPS} PROFILE_N_OPS=${PROFILE_N_OPS} DO_CORRECTNESS=${DO_CORRECTNESS}"

for target in "${targets[@]}"; do
  run_one "${target}" baseline 100
  run_one "${target}" tuned "$(tuned_bounds_for_target "${target}")"
done

python3 summarize_profile_compare.py "${out_dir}" | tee "${out_dir}/summarize_profile_compare.log"

echo
echo "[done] ${out_dir}"
echo "[show] runs:"
cat "${runs_csv}"
echo
echo "[show] key kernel compare:"
cat "${out_dir}/key_kernel_compare.csv"
