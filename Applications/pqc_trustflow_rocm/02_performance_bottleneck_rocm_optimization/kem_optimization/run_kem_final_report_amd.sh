#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="amd_results/final_report_${stamp}"
mkdir -p "${out_dir}"

meta="${out_dir}/environment.txt"
{
  echo "timestamp=${stamp}"
  echo "pwd=$(pwd)"
  echo
  echo "== hipcc --version =="
  hipcc --version || true
  echo
  echo "== tools =="
  which hipcc || true
  which rocprofv3 || true
  which rocm-smi || true
  echo
  echo "== rocm-smi =="
  rocm-smi --showproductname --showdriverversion --showvbios --showmeminfo vram || true
} 2>&1 | tee "${meta}"

summary="${out_dir}/kem_final_summary.log"
: > "${summary}"

run_one() {
  local target="$1"
  local batch="$2"
  local iters="$3"
  local kg_tpb="$4"
  local enc_tpb="$5"
  local dec_tpb="$6"
  local keypair_bounds="${7:-1}"
  local encaps_bounds="${8:-0}"
  local decaps_bounds="${9:-0}"
  local log="${out_dir}/${target}_b${batch}_n${iters}_tpb${kg_tpb}_${enc_tpb}_${dec_tpb}_bounds${keypair_bounds}${encaps_bounds}${decaps_bounds}.log"

  {
    echo "========== ${target} =========="
    echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "batch=${batch} n_ops=${iters} KEM_KEYGEN_TPB=${kg_tpb} KEM_ENCAPS_TPB=${enc_tpb} KEM_DECAPS_TPB=${dec_tpb}"
    echo "KEM_KEYPAIR_LAUNCH_BOUNDS=${keypair_bounds} KEM_ENCAPS_LAUNCH_BOUNDS=${encaps_bounds} KEM_DECAPS_LAUNCH_BOUNDS=${decaps_bounds}"
    KEM_KEYGEN_TPB="${kg_tpb}" KEM_ENCAPS_TPB="${enc_tpb}" KEM_DECAPS_TPB="${dec_tpb}" \
    KEM_KEYPAIR_LAUNCH_BOUNDS="${keypair_bounds}" KEM_ENCAPS_LAUNCH_BOUNDS="${encaps_bounds}" KEM_DECAPS_LAUNCH_BOUNDS="${decaps_bounds}" \
      bash build_hip.sh "${target}"
    "./${target}_amd" --batch "${batch}" --n-ops "${iters}" --no-correctness
    echo
  } 2>&1 | tee "${log}" | tee -a "${summary}"
}

# Stable final KEM table. Kyber uses batch 32768, Aigis-enc uses batch 65536.
# Bounds are selected from the 2026-06-14 all-parameter bounds probe using the
# balanced score: 0.30*keygen + 0.40*encaps + 0.30*decaps.
run_one kyber512 32768 20 256 128 128 0 0 1
run_one kyber768 32768 20 256 128 128 0 1 0
run_one kyber1024 32768 20 256 128 128 1 1 0

run_one aigisenc1 65536 20 256 128 128 1 0 1
run_one aigisenc2 65536 20 256 128 128 1 1 0
run_one aigisenc3 65536 20 256 128 128 1 0 1
run_one aigisenc4 65536 20 256 128 128 1 0 1

echo "[extract] ${out_dir}/kem_final_extract.txt"
grep -E "Algorithm:|Keygen:|Encaps:|Decaps:" "${summary}" | tee "${out_dir}/kem_final_extract.txt"

echo "[done] final report directory: ${out_dir}"
