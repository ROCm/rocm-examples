#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

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
  -DBATCH_SIGN_DECOMP_TAIL_ENABLE=0
  -DBATCH_SIGN_DECOMP_ADAPTIVE_ENABLE=0
  -DBATCH_SIGN_CP_FUSE_ENABLE=0
  -DBATCH_SIGN_SAMPLE_DUP_YHAT=0
  -DBATCH_KEYGEN_SAMPLE_SPLIT_FAST=1
)

mkdir -p amd_results/build

build_one() {
  local alg="$1"
  local mode="$2"
  local out="$3"
  echo "[build] ${out}"
  hipcc "${COMMON[@]}" -DALGORITHM="${alg}" -DPARAM_MODE="${mode}" main.cu -o "${out}" \
    2>&1 | tee "amd_results/build/${out}.log"
}

build_one 1 2 mldsa44_amd
build_one 1 3 mldsa65_amd
build_one 1 5 mldsa87_amd
build_one 2 1 aigis1_amd
build_one 2 2 aigis2_amd
build_one 2 3 aigis3_amd

echo "[build] done"
