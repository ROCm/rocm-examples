#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

plan="${1:-amd_results/sig_amd_variant_plan.env}"
if [[ -f "${plan}" ]]; then
  # shellcheck disable=SC1090
  source "${plan}"
  echo "[select] loaded ${plan}"
else
  echo "[select] ${plan} not found; using base for all targets"
fi

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

target_variant() {
  case "$1" in
    mldsa44) echo "${SIG_AMD_VARIANT_MLDSA44:-base}" ;;
    mldsa65) echo "${SIG_AMD_VARIANT_MLDSA65:-base}" ;;
    mldsa87) echo "${SIG_AMD_VARIANT_MLDSA87:-base}" ;;
    aigis1)  echo "${SIG_AMD_VARIANT_AIGIS1:-base}" ;;
    aigis2)  echo "${SIG_AMD_VARIANT_AIGIS2:-base}" ;;
    aigis3)  echo "${SIG_AMD_VARIANT_AIGIS3:-base}" ;;
    *)
      echo "unknown target: $1" >&2
      return 1
      ;;
  esac
}

mkdir -p amd_results/build

build_one() {
  local alg="$1"
  local mode="$2"
  local target="$3"
  local out="$4"
  local variant extra
  variant="$(target_variant "${target}")"
  extra="$(variant_flags "${variant}")"

  echo "[build] ${out} target=${target} variant=${variant}"
  # shellcheck disable=SC2086
  hipcc "${COMMON[@]}" ${extra} -DALGORITHM="${alg}" -DPARAM_MODE="${mode}" main.cu -o "${out}" \
    2>&1 | tee "amd_results/build/${out}.log"
}

build_one 1 2 mldsa44 mldsa44_amd
build_one 1 3 mldsa65 mldsa65_amd
build_one 1 5 mldsa87 mldsa87_amd
build_one 2 1 aigis1  aigis1_amd
build_one 2 2 aigis2  aigis2_amd
build_one 2 3 aigis3  aigis3_amd

echo "[build] selected variants done"
