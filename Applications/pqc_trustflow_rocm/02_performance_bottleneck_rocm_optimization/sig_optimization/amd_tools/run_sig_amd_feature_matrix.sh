#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

out_dir="amd_results/amd_feature_matrix"
mkdir -p "${out_dir}" "amd_results/build"

COMMON=(
  -O2
  -std=c++17
  -x hip
  --offload-arch=gfx1100
  -DBLOCK_SIZE=1
  -DBATCH_KEYGEN_INTERNAL_MATERIAL=1
  -DBATCH_SIGN_WARP_ENABLE=0
  -DBATCH_SIGN_MONO_ENABLE=0
  -DBATCH_SIGN_PRECOMP_REUSE=0
  -DBATCH_SIGN_LARGE_STRATEGY_ENABLE=0
  -DBATCH_SIGN_DECOMP_ENABLE=1
  -DBATCH_KEYGEN_SAMPLE_SPLIT_FAST=1
)

# Representative default targets. Override with:
#   FEATURE_TARGETS="mldsa44 mldsa87 aigis2"
#   FEATURE_BATCHES="1024 8192 16384"
#   FEATURE_MODES="independent paper"
read -r -a target_names <<< "${FEATURE_TARGETS:-mldsa44 mldsa87 aigis2}"
read -r -a batches <<< "${FEATURE_BATCHES:-1024 8192}"
read -r -a modes <<< "${FEATURE_MODES:-independent paper}"
read -r -a variants <<< "${FEATURE_VARIANTS:-base adaptive check8 check16 wave64_ctrl cp_fuse tail16_base tail16_cp_fuse yhat_dup}"
repeats="${FEATURE_REPEATS:-1}"

target_alg_mode() {
  case "$1" in
    mldsa44) echo "1 2" ;;
    mldsa65) echo "1 3" ;;
    mldsa87) echo "1 5" ;;
    aigis1)  echo "2 1" ;;
    aigis2)  echo "2 2" ;;
    aigis3)  echo "2 3" ;;
    *)
      echo "unknown target: $1" >&2
      return 1
      ;;
  esac
}

variant_flags() {
  case "$1" in
    base)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0 -DBATCH_SIGN_CP_FUSE_ENABLE=0 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=0 -DBATCH_SIGN_SAMPLE_DUP_YHAT=0 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=4"
      ;;
    adaptive)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=1 -DBATCH_SIGN_CP_FUSE_ENABLE=0 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=0 -DBATCH_SIGN_SAMPLE_DUP_YHAT=0 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=4"
      ;;
    cp_fuse)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0 -DBATCH_SIGN_CP_FUSE_ENABLE=1 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=0 -DBATCH_SIGN_SAMPLE_DUP_YHAT=0 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=4"
      ;;
    check8)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0 -DBATCH_SIGN_CP_FUSE_ENABLE=0 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=0 -DBATCH_SIGN_SAMPLE_DUP_YHAT=0 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=8"
      ;;
    check16)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0 -DBATCH_SIGN_CP_FUSE_ENABLE=0 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=0 -DBATCH_SIGN_SAMPLE_DUP_YHAT=0 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=16"
      ;;
    wave64_ctrl)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0 -DBATCH_SIGN_CP_FUSE_ENABLE=0 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=0 -DBATCH_SIGN_SAMPLE_DUP_YHAT=0 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=4 -DBATCH_SIGN_HASH_TPB=64 -DBATCH_SIGN_CHECK_TPB=64"
      ;;
    wave64_check8)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0 -DBATCH_SIGN_CP_FUSE_ENABLE=0 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=0 -DBATCH_SIGN_SAMPLE_DUP_YHAT=0 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=8 -DBATCH_SIGN_HASH_TPB=64 -DBATCH_SIGN_CHECK_TPB=64"
      ;;
    tail16_base)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0 -DBATCH_SIGN_CP_FUSE_ENABLE=0 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=1 -DBATCH_SIGN_DECOMP_TAIL_AFTER=16 -DBATCH_SIGN_DECOMP_TAIL_PENDING_DIV=256 -DBATCH_SIGN_DECOMP_TAIL_PENDING_MIN=8 -DBATCH_SIGN_SAMPLE_DUP_YHAT=0 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=4"
      ;;
    tail16_cp_fuse|tail16)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0 -DBATCH_SIGN_CP_FUSE_ENABLE=1 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=1 -DBATCH_SIGN_DECOMP_TAIL_AFTER=16 -DBATCH_SIGN_DECOMP_TAIL_PENDING_DIV=256 -DBATCH_SIGN_DECOMP_TAIL_PENDING_MIN=8 -DBATCH_SIGN_SAMPLE_DUP_YHAT=0 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=4"
      ;;
    yhat_dup)
      echo "-DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0 -DBATCH_SIGN_CP_FUSE_ENABLE=1 -DBATCH_SIGN_DECOMP_TAIL_ENABLE=0 -DBATCH_SIGN_SAMPLE_DUP_YHAT=1 -DBATCH_SIGN_DECOMP_CHECK_INTERVAL=4"
      ;;
    *)
      echo "unknown variant: $1" >&2
      return 1
      ;;
  esac
}

build_one() {
  local target="$1"
  local variant="$2"
  local alg mode extra exe
  read -r alg mode <<< "$(target_alg_mode "${target}")"
  extra="$(variant_flags "${variant}")"
  exe="${target}_${variant}_amd"
  echo "[build] ${exe}"
  # shellcheck disable=SC2086
  hipcc "${COMMON[@]}" ${extra} -DALGORITHM="${alg}" -DPARAM_MODE="${mode}" main.cu -o "${exe}" \
    2>&1 | tee "amd_results/build/${exe}.log"
}

run_one() {
  local target="$1"
  local variant="$2"
  local mode="$3"
  local batch="$4"
  local repeat="$5"
  local exe="${target}_${variant}_amd"
  local mode_flag="--bench-${mode}"
  local log
  if [[ "${repeats}" -gt 1 ]]; then
    log="${out_dir}/${target}_${variant}_${mode}_b${batch}_r${repeat}.log"
  else
    log="${out_dir}/${target}_${variant}_${mode}_b${batch}.log"
  fi

  if [[ ! -x "./${exe}" ]]; then
    echo "[skip] ./${exe} not found or not executable"
    return 0
  fi

  echo "[feature] ${exe} mode=${mode} batch=${batch} repeat=${repeat}/${repeats}"
  set +e
  stdbuf -oL -eL "./${exe}" "${mode_flag}" --batch "${batch}" --quiet --skip-keygen-oracle \
    2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  echo "[feature] exit_code=${rc}" | tee -a "${log}"
}

for target in "${target_names[@]}"; do
  for variant in "${variants[@]}"; do
    build_one "${target}" "${variant}"
  done
done

for target in "${target_names[@]}"; do
  for mode in "${modes[@]}"; do
    for batch in "${batches[@]}"; do
      for repeat in $(seq 1 "${repeats}"); do
        for variant in "${variants[@]}"; do
          run_one "${target}" "${variant}" "${mode}" "${batch}" "${repeat}"
        done
      done
    done
  done
done

python3 amd_tools/parse_sig_results.py "${out_dir}" > amd_results/sig_amd_feature_matrix.csv
python3 amd_tools/summarize_amd_feature_matrix.py amd_results/sig_amd_feature_matrix.csv \
  > amd_results/sig_amd_feature_matrix_ranked.csv

echo "[summary] amd_results/sig_amd_feature_matrix.csv"
echo "[summary] amd_results/sig_amd_feature_matrix_ranked.csv"
