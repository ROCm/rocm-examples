#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

exe="${1:-mldsa44_amd}"
batch="${2:-1024}"
mkdir -p amd_results/profile

if [[ ! -x "./${exe}" ]]; then
  echo "error: ./${exe} not found or not executable" >&2
  exit 1
fi

plain_log="amd_results/profile/${exe}_b${batch}_profile.log"
rocprof_dir="amd_results/profile/${exe}_b${batch}_rocprof"

echo "[profile] app-level profile: ${exe} batch=${batch}"
stdbuf -oL -eL "./${exe}" --batch "${batch}" --quiet --skip-keygen-oracle --profile \
  2>&1 | tee "${plain_log}"

if command -v rocprofv3 >/dev/null 2>&1; then
  echo "[profile] rocprofv3 output: ${rocprof_dir}"
  rm -rf "${rocprof_dir}"
  mkdir -p "${rocprof_dir}"
  rocprof_log="amd_results/profile/${exe}_b${batch}_rocprof.log"
  set +e
  rocprofv3 --output-directory "${rocprof_dir}" -- \
    "./${exe}" --batch "${batch}" --quiet --skip-keygen-oracle \
    2>&1 | tee "${rocprof_log}"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "[profile] rocprofv3 failed with exit_code=${rc}; see ${rocprof_log}" >&2
    exit "${rc}"
  fi
else
  echo "[profile] rocprofv3 not found; skipped ROCm trace"
fi
