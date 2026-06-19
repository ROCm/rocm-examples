#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

TARGETS=${TARGETS:-"kyber768 kyber1024 aigisenc4"}
N_OPS=${N_OPS:-20}
PROFILE_N_OPS=${PROFILE_N_OPS:-1}
ENABLE_SYS_TRACE=${ENABLE_SYS_TRACE:-1}
ENABLE_PMC=${ENABLE_PMC:-0}
ENABLE_SMI=${ENABLE_SMI:-1}
TOOL_TIMEOUT=${TOOL_TIMEOUT:-120}
KYBER_BATCH=${KYBER_BATCH:-32768}
AIGIS_BATCH=${AIGIS_BATCH:-65536}
KEM_KEYGEN_TPB=${KEM_KEYGEN_TPB:-256}
KEM_ENCAPS_TPB=${KEM_ENCAPS_TPB:-128}
KEM_DECAPS_TPB=${KEM_DECAPS_TPB:-128}
OPT_LEVEL=${OPT_LEVEL:-O2}

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="amd_results/rocm_toolbox_${stamp}"
mkdir -p "${out_dir}"

tool_log="${out_dir}/tool_discovery.txt"
{
  echo "timestamp=${stamp}"
  echo "pwd=$(pwd)"
  echo "TARGETS=${TARGETS}"
  echo "N_OPS=${N_OPS}"
  echo "PROFILE_N_OPS=${PROFILE_N_OPS}"
  echo "ENABLE_SYS_TRACE=${ENABLE_SYS_TRACE}"
  echo "ENABLE_PMC=${ENABLE_PMC}"
  echo "ENABLE_SMI=${ENABLE_SMI}"
  echo "TOOL_TIMEOUT=${TOOL_TIMEOUT}"
  echo
  echo "== command availability =="
  for t in hipcc rocprofv3 rocprof-compute rocm-smi rocminfo hipconfig llvm-objdump; do
    printf "%-18s" "${t}"
    command -v "${t}" || true
  done
  echo
  echo "== hipcc --version =="
  hipcc --version || true
  echo
  echo "== hipconfig =="
  hipconfig || true
  echo
  echo "== rocminfo head =="
  rocminfo 2>/dev/null | head -120 || true
  echo
  echo "== rocm-smi static =="
  rocm-smi --showproductname --showdriverversion --showvbios --showmeminfo vram --showclocks --showmaxpower || true
} > "${tool_log}" 2>&1

if command -v rocprofv3 >/dev/null 2>&1; then
  rocprofv3 --list-avail > "${out_dir}/rocprofv3_list_avail.txt" 2>&1 || true
fi

run_with_timeout() {
  local seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "${seconds}" "$@"
  else
    "$@"
  fi
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

candidate_counters=(
  SQ_WAVES
  GRBM_GUI_ACTIVE
  GPUBusy
  VALUUtilization
  VALUBusy
  SALUBusy
  MemUnitBusy
  MemUnitStalled
  FetchSize
  WriteSize
  FETCH_SIZE
  WRITE_SIZE
  LDSBankConflict
  CU_OCCUPANCY
  MeanOccupancyPerCU
  MeanOccupancyPerActiveCU
)

select_counters() {
  local list_file="${out_dir}/rocprofv3_list_avail.txt"
  local selected=()
  [[ -f "${list_file}" ]] || return 0
  for c in "${candidate_counters[@]}"; do
    if grep -qw "${c}" "${list_file}"; then
      selected+=("${c}")
    fi
  done
  local joined=""
  for c in "${selected[@]}"; do
    if [[ -z "${joined}" ]]; then
      joined="${c}"
    else
      joined="${joined},${c}"
    fi
  done
  echo "${joined}"
}

extract_metric() {
  local label="$1"
  local log="$2"
  grep -E "  ${label}:" "${log}" \
    | tail -1 \
    | grep -oE '[0-9]+ ops/sec' \
    | tail -1 \
    | awk '{print $1}'
}

runs_csv="${out_dir}/toolbox_runs.csv"
echo "target,bounds,batch,n_ops,keygen_ops_s,encaps_ops_s,decaps_ops_s,status,run_dir,counters" > "${runs_csv}"

run_target() {
  local target="$1"
  local bounds
  bounds="$(tuned_bounds_for_target "${target}")"
  local kb="${bounds:0:1}"
  local eb="${bounds:1:1}"
  local db="${bounds:2:1}"
  local batch
  batch="$(batch_for_target "${target}")"
  local run_dir="${out_dir}/${target}_bounds${bounds}"
  mkdir -p "${run_dir}"
  local status="PASS"

  echo
  echo "[toolbox] target=${target} bounds=${bounds} batch=${batch}"

  OPT_LEVEL="${OPT_LEVEL}" \
  KEM_KEYGEN_TPB="${KEM_KEYGEN_TPB}" KEM_ENCAPS_TPB="${KEM_ENCAPS_TPB}" KEM_DECAPS_TPB="${KEM_DECAPS_TPB}" \
  KEM_KEYPAIR_LAUNCH_BOUNDS="${kb}" KEM_ENCAPS_LAUNCH_BOUNDS="${eb}" KEM_DECAPS_LAUNCH_BOUNDS="${db}" \
    bash build_hip.sh "${target}" > "${run_dir}/build.log" 2>&1 || status="FAIL"

  if [[ "${status}" == "PASS" ]]; then
    local smi_pid=""
    if [[ "${ENABLE_SMI}" == "1" ]]; then
      (
        for i in $(seq 1 80); do
          echo "===== sample ${i} $(date '+%H:%M:%S.%3N') ====="
          rocm-smi --showuse --showmemuse --showtemp --showpower --showclocks
          sleep 0.2
        done
      ) > "${run_dir}/rocm_smi_during.log" &
      smi_pid=$!
    else
      echo "SMI sampling disabled." > "${run_dir}/rocm_smi_during.log"
    fi

    run_with_timeout "${TOOL_TIMEOUT}" "./${target}_amd" --batch "${batch}" --n-ops "${N_OPS}" --no-correctness \
      > "${run_dir}/benchmark.log" 2>&1 || status="FAIL"
    if [[ -n "${smi_pid}" ]]; then
      wait "${smi_pid}" || true
    fi
  fi

  local kg_ops=""
  local enc_ops=""
  local dec_ops=""
  if [[ "${status}" == "PASS" ]]; then
    kg_ops="$(extract_metric Keygen "${run_dir}/benchmark.log" || true)"
    enc_ops="$(extract_metric Encaps "${run_dir}/benchmark.log" || true)"
    dec_ops="$(extract_metric Decaps "${run_dir}/benchmark.log" || true)"
  fi

  if [[ "${status}" == "PASS" && -x "./${target}_amd" && "$(command -v rocprofv3 || true)" ]]; then
    mkdir -p "${run_dir}/sys_trace"
    if [[ "${ENABLE_SYS_TRACE}" == "1" ]]; then
      run_with_timeout "${TOOL_TIMEOUT}" rocprofv3 \
        --sys-trace \
        --output-format csv \
        --output-directory "${run_dir}/sys_trace" \
        -- \
        "./${target}_amd" --batch "${batch}" --n-ops "${PROFILE_N_OPS}" --no-correctness \
        > "${run_dir}/rocprofv3_sys_trace.log" 2>&1 || true
    else
      echo "sys-trace disabled." > "${run_dir}/rocprofv3_sys_trace.log"
    fi

    local counters
    counters="$(select_counters)"
    if [[ "${ENABLE_PMC}" == "1" && -n "${counters}" ]]; then
      echo "${counters}" > "${run_dir}/selected_counters.txt"
      mkdir -p "${run_dir}/pmc"
      run_with_timeout "${TOOL_TIMEOUT}" rocprofv3 \
        --pmc "${counters}" \
        --output-format csv \
        --output-directory "${run_dir}/pmc" \
        -- \
        "./${target}_amd" --batch "${batch}" --n-ops "${PROFILE_N_OPS}" --no-correctness \
        > "${run_dir}/rocprofv3_pmc.log" 2>&1 || true
    elif [[ "${ENABLE_PMC}" != "1" ]]; then
      echo "PMC disabled because ENABLE_PMC=${ENABLE_PMC}. Enable explicitly with ENABLE_PMC=1." > "${run_dir}/selected_counters.txt"
    else
      echo "No candidate counters found in rocprofv3 --list-avail output." > "${run_dir}/selected_counters.txt"
    fi
  fi

  echo "${target},${bounds},${batch},${N_OPS},${kg_ops},${enc_ops},${dec_ops},${status},${target}_bounds${bounds},$(cat "${run_dir}/selected_counters.txt" 2>/dev/null || true)" | tee -a "${runs_csv}"
}

for target in ${TARGETS}; do
  run_target "${target}"
done

python3 summarize_rocm_pmc.py "${out_dir}" | tee "${out_dir}/summarize_rocm_pmc.log" || true

echo
echo "[done] ${out_dir}"
echo "[show] toolbox runs:"
cat "${runs_csv}"
echo
echo "[show] pmc summary:"
cat "${out_dir}/pmc_summary.csv" 2>/dev/null || true
