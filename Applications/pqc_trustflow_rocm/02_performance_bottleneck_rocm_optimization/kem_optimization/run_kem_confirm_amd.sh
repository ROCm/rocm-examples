#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

TARGET=${1:-kyber768}
BATCH=${BATCH:-32768}
N_OPS=${N_OPS:-50}
REPEATS=${REPEATS:-3}

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="amd_results/confirm_${TARGET}_${stamp}"
mkdir -p "${out_dir}"

summary="${out_dir}/confirm_summary.csv"
echo "target,tag,repeat,batch,n_ops,opt,kg_tpb,enc_tpb,dec_tpb,keypair_bounds,encaps_bounds,decaps_bounds,keygen_ops_s,encaps_ops_s,decaps_ops_s,status,log" > "${summary}"

extract_metric() {
  local label="$1"
  local log="$2"
  grep -E "  ${label}:" "${log}" \
    | tail -1 \
    | grep -oE '[0-9]+ ops/sec' \
    | tail -1 \
    | awk '{print $1}'
}

run_candidate() {
  local tag="$1"
  local opt="$2"
  local kg="$3"
  local enc="$4"
  local dec="$5"
  local kb="$6"
  local eb="$7"
  local db="$8"

  for rep in $(seq 1 "${REPEATS}"); do
    local log="${out_dir}/${TARGET}_${tag}_r${rep}.log"
    local status="PASS"
    {
      echo "========== ${TARGET} ${tag} repeat=${rep}/${REPEATS} =========="
      echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "batch=${BATCH} n_ops=${N_OPS}"
      echo "OPT_LEVEL=${opt} KEM_KEYGEN_TPB=${kg} KEM_ENCAPS_TPB=${enc} KEM_DECAPS_TPB=${dec}"
      echo "bounds=${kb}/${eb}/${db}"
      OPT_LEVEL="${opt}" \
      KEM_KEYGEN_TPB="${kg}" KEM_ENCAPS_TPB="${enc}" KEM_DECAPS_TPB="${dec}" \
      KEM_KEYPAIR_LAUNCH_BOUNDS="${kb}" KEM_ENCAPS_LAUNCH_BOUNDS="${eb}" KEM_DECAPS_LAUNCH_BOUNDS="${db}" \
        bash build_hip.sh "${TARGET}"
      "./${TARGET}_amd" --batch 128 --n-ops 1
      "./${TARGET}_amd" --batch "${BATCH}" --n-ops "${N_OPS}" --no-correctness
    } > "${log}" 2>&1 || status="FAIL"

    local kg_ops=""
    local enc_ops=""
    local dec_ops=""
    if [[ "${status}" == "PASS" ]]; then
      kg_ops="$(extract_metric Keygen "${log}" || true)"
      enc_ops="$(extract_metric Encaps "${log}" || true)"
      dec_ops="$(extract_metric Decaps "${log}" || true)"
    fi

    echo "${TARGET},${tag},${rep},${BATCH},${N_OPS},${opt},${kg},${enc},${dec},${kb},${eb},${db},${kg_ops},${enc_ops},${dec_ops},${status},${log}" | tee -a "${summary}"
  done
}

echo "[confirm] target=${TARGET} batch=${BATCH} n_ops=${N_OPS} repeats=${REPEATS} out=${out_dir}"

# Current stable baseline from the previous final report.
run_candidate baseline_o2_256_128_128_b100 O2 256 128 128 1 0 0

# Best balanced candidate from the first Kyber-768 tuning pass:
# encaps launch bounds improves encaps while keygen/decaps stay close to baseline.
run_candidate balanced_encbounds_o2_b010 O2 256 128 128 0 1 0

# Slightly higher encaps candidate in the first pass, kept separate to check
# whether keypair launch bounds changes repeat-to-repeat stability.
run_candidate encbest_o2_b110 O2 256 128 128 1 1 0

# Best keygen candidate observed in the first pass.
run_candidate keygenbest_o3_b100 O3 256 128 128 1 0 0

# Best decaps candidate observed in the first pass.
run_candidate decbest_o3_kg512_b100 O3 512 128 128 1 0 0

echo
echo "[done] summary=${summary}"
echo "[hint] show summary:"
echo "cat ${summary}"
echo "[hint] rank encaps:"
echo "sort -t, -k14,14nr ${summary} | head"
