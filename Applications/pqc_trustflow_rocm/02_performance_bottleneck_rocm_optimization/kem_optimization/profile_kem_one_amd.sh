#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

exe="${1:-kyber768_amd}"
batch="${2:-8192}"
iters="${3:-3}"

mkdir -p amd_results/profile

if [[ ! -x "./${exe}" ]]; then
  echo "error: ./${exe} not found or not executable" >&2
  exit 1
fi

plain_log="amd_results/profile/${exe}_b${batch}_profile.log"
rocprof_dir="amd_results/profile/${exe}_b${batch}_rocprof"

echo "[profile] app-level pipeline profile: ${exe} batch=${batch}"
stdbuf -oL -eL "./${exe}" --batch "${batch}" --n-ops "${iters}" \
  --no-correctness --pipeline --profile-pipeline \
  2>&1 | tee "${plain_log}"

if command -v rocprofv3 >/dev/null 2>&1; then
  echo "[profile] rocprofv3 output: ${rocprof_dir}"
  rm -rf "${rocprof_dir}"
  mkdir -p "${rocprof_dir}"
  rocprofv3 --output-directory "${rocprof_dir}" --timestamp on -- \
    "./${exe}" --batch "${batch}" --n-ops 1 --no-correctness --pipeline \
    2>&1 | tee "amd_results/profile/${exe}_b${batch}_rocprof.log"
else
  echo "[profile] rocprofv3 not found; skipped ROCm trace"
fi
