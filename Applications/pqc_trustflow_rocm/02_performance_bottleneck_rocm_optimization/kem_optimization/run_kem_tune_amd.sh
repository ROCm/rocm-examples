#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

TARGET=${1:-kyber768}
BATCH=${BATCH:-32768}
N_OPS=${N_OPS:-20}
DO_CORRECTNESS=${DO_CORRECTNESS:-1}

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="amd_results/tune_${TARGET}_${stamp}"
mkdir -p "${out_dir}"

summary="${out_dir}/tune_summary.csv"
echo "target,batch,n_ops,opt,kg_tpb,enc_tpb,dec_tpb,keypair_bounds,encaps_bounds,decaps_bounds,wp_kg_warps,pack_tpb,keygen_ops_s,encaps_ops_s,decaps_ops_s,status,log" > "${summary}"

extract_metric() {
  local label="$1"
  local log="$2"
  grep -E "  ${label}:" "${log}" \
    | tail -1 \
    | grep -oE '[0-9]+ ops/sec' \
    | tail -1 \
    | awk '{print $1}'
}

run_config() {
  local opt="$1"
  local kg="$2"
  local enc="$3"
  local dec="$4"
  local kb="$5"
  local eb="$6"
  local db="$7"
  local wp="$8"
  local pack="$9"
  local tag="opt${opt}_kg${kg}_enc${enc}_dec${dec}_b${kb}${eb}${db}_wp${wp}_pack${pack}"
  local log="${out_dir}/${TARGET}_${tag}.log"
  local status="PASS"

  {
    echo "========== ${TARGET} ${tag} =========="
    echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "batch=${BATCH} n_ops=${N_OPS}"
    echo "OPT_LEVEL=${opt} KEM_KEYGEN_TPB=${kg} KEM_ENCAPS_TPB=${enc} KEM_DECAPS_TPB=${dec}"
    echo "bounds=${kb}/${eb}/${db} WP_KG_WARPS_BLOCK=${wp} KEM_PACK_TPB=${pack}"
    OPT_LEVEL="${opt}" \
    KEM_KEYGEN_TPB="${kg}" KEM_ENCAPS_TPB="${enc}" KEM_DECAPS_TPB="${dec}" \
    KEM_KEYPAIR_LAUNCH_BOUNDS="${kb}" KEM_ENCAPS_LAUNCH_BOUNDS="${eb}" KEM_DECAPS_LAUNCH_BOUNDS="${db}" \
    WP_KG_WARPS_BLOCK="${wp}" KEM_PACK_TPB="${pack}" \
      bash build_hip.sh "${TARGET}"

    if [[ "${DO_CORRECTNESS}" == "1" ]]; then
      "./${TARGET}_amd" --batch 128 --n-ops 1
    fi

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

  echo "${TARGET},${BATCH},${N_OPS},${opt},${kg},${enc},${dec},${kb},${eb},${db},${wp},${pack},${kg_ops},${enc_ops},${dec_ops},${status},${log}" | tee -a "${summary}"
}

echo "[tune] target=${TARGET} batch=${BATCH} n_ops=${N_OPS} out=${out_dir}"

# Serial-path tuning. This is the current final-report path and therefore the
# first optimization surface to lock down.
for opt in O2 O3; do
  for kg in 128 256 512; do
    for enc in 64 128; do
      for dec in 64 128; do
        run_config "${opt}" "${kg}" "${enc}" "${dec}" 1 0 0 4 128
      done
    done
  done
done

# Targeted launch-bounds checks for the current best neighborhood.
for kb in 0 1; do
  for eb in 0 1; do
    for db in 0 1; do
      run_config O2 256 128 128 "${kb}" "${eb}" "${db}" 4 128
    done
  done
done

# Pipeline-only knobs. These runs still print serial first, but the added
# pipeline profile makes it easy to see whether the sampling path improves.
pipeline_log="${out_dir}/pipeline_candidates.log"
: > "${pipeline_log}"
for wp in 2 4 8; do
  for pack in 64 128 256; do
    tag="pipeline_wp${wp}_pack${pack}"
    log="${out_dir}/${TARGET}_${tag}.log"
    {
      echo "========== ${TARGET} ${tag} =========="
      OPT_LEVEL=O2 KEM_KEYGEN_TPB=256 KEM_ENCAPS_TPB=128 KEM_DECAPS_TPB=128 \
      WP_KG_WARPS_BLOCK="${wp}" KEM_PACK_TPB="${pack}" \
        bash build_hip.sh "${TARGET}"
      "./${TARGET}_amd" --batch "${BATCH}" --n-ops 3 --no-correctness --pipeline --profile-pipeline
    } > "${log}" 2>&1 || true
    grep -E "Algorithm:|Pipeline profile:|Keygen:|Encaps:|Decaps:" "${log}" >> "${pipeline_log}" || true
    echo >> "${pipeline_log}"
  done
done

echo
echo "[done] summary=${summary}"
echo "[done] pipeline=${pipeline_log}"
echo "[hint] sort by keygen:  sort -t, -k13,13nr ${summary} | head"
echo "[hint] sort by encaps: sort -t, -k14,14nr ${summary} | head"
echo "[hint] sort by decaps: sort -t, -k15,15nr ${summary} | head"
