#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

target="${1:-kyber768}"
batch="${2:-32768}"
iters="${3:-200}"
kg_tpb="${KEM_KEYGEN_TPB:-256}"
enc_tpb="${KEM_ENCAPS_TPB:-128}"
dec_tpb="${KEM_DECAPS_TPB:-128}"
stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="amd_results/resource_profile_${target}_${stamp}"
mkdir -p "${out_dir}"

{
  echo "timestamp=${stamp}"
  echo "target=${target}"
  echo "batch=${batch}"
  echo "iters=${iters}"
  echo "KEM_KEYGEN_TPB=${kg_tpb}"
  echo "KEM_ENCAPS_TPB=${enc_tpb}"
  echo "KEM_DECAPS_TPB=${dec_tpb}"
  hipcc --version || true
} 2>&1 | tee "${out_dir}/metadata.txt"

KEM_KEYGEN_TPB="${kg_tpb}" KEM_ENCAPS_TPB="${enc_tpb}" KEM_DECAPS_TPB="${dec_tpb}" \
  bash build_hip.sh "${target}" 2>&1 | tee "${out_dir}/build.log"

(
  for i in $(seq 1 120); do
    echo "===== sample ${i} $(date '+%H:%M:%S.%3N') ====="
    rocm-smi --showuse --showmemuse --showtemp --showpower
    sleep 0.2
  done
) > "${out_dir}/rocm_smi_during.log" &
smi_pid=$!

"./${target}_amd" --batch "${batch}" --n-ops "${iters}" --no-correctness \
  2>&1 | tee "${out_dir}/benchmark.log"

wait "${smi_pid}" || true

mkdir -p "${out_dir}/rocprofv3"
rocprofv3 \
  --kernel-trace \
  --hip-trace \
  --output-format csv \
  --output-directory "${out_dir}/rocprofv3" \
  -- \
  "./${target}_amd" --batch "${batch}" --n-ops 1 --no-correctness --pipeline \
  2>&1 | tee "${out_dir}/rocprofv3.log" || true

python3 summarize_rocprofv3_trace.py "${out_dir}/rocprofv3" \
  > "${out_dir}/rocprofv3_summary.csv" || true

grep -E "GPU\\[0\\].*(GPU use|VRAM|Power|Temperature)" "${out_dir}/rocm_smi_during.log" \
  > "${out_dir}/rocm_smi_gpu0_extract.log" || true

echo "[done] resource profile directory: ${out_dir}"
