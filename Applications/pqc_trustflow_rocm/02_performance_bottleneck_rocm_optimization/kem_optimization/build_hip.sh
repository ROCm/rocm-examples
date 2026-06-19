#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

HIPCC=${HIPCC:-hipcc}
ROCM_ARCH=${ROCM_ARCH:-gfx1100}
KEM_SERIAL_TPB=${KEM_SERIAL_TPB:-64}
KEM_KEYGEN_TPB=${KEM_KEYGEN_TPB:-${KEM_SERIAL_TPB}}
KEM_ENCAPS_TPB=${KEM_ENCAPS_TPB:-${KEM_SERIAL_TPB}}
KEM_DECAPS_TPB=${KEM_DECAPS_TPB:-${KEM_SERIAL_TPB}}
KEM_KEYPAIR_LAUNCH_BOUNDS=${KEM_KEYPAIR_LAUNCH_BOUNDS:-1}
KEM_ENCAPS_LAUNCH_BOUNDS=${KEM_ENCAPS_LAUNCH_BOUNDS:-}
KEM_DECAPS_LAUNCH_BOUNDS=${KEM_DECAPS_LAUNCH_BOUNDS:-}
WP_KG_WARPS_BLOCK=${WP_KG_WARPS_BLOCK:-4}
KEM_PACK_TPB=${KEM_PACK_TPB:-128}
BUILD_TYPE=${BUILD_TYPE:-Release}
CXX_STD=${CXX_STD:-c++17}
ROCM_WAVE32_FLAG=${ROCM_WAVE32_FLAG:-}
OPT_LEVEL=${OPT_LEVEL:-}
EXTRA_HIPCC_FLAGS=${EXTRA_HIPCC_FLAGS:-}

if [[ "${BUILD_TYPE}" == "Debug" ]]; then
  OPT_FLAGS=(-O0 -g)
else
  if [[ -n "${OPT_LEVEL}" ]]; then
    OPT_FLAGS=("-${OPT_LEVEL}")
  else
    OPT_FLAGS=(-O2)
  fi
fi

COMMON_FLAGS=(
  "${OPT_FLAGS[@]}"
  -std="${CXX_STD}"
  -x
  hip
  --offload-arch="${ROCM_ARCH}"
  -DKEM_SERIAL_TPB="${KEM_SERIAL_TPB}"
  -DKEM_KEYGEN_TPB="${KEM_KEYGEN_TPB}"
  -DKEM_ENCAPS_TPB="${KEM_ENCAPS_TPB}"
  -DKEM_DECAPS_TPB="${KEM_DECAPS_TPB}"
  -DKEM_KEYPAIR_LAUNCH_BOUNDS="${KEM_KEYPAIR_LAUNCH_BOUNDS}"
  -DWP_KG_WARPS_BLOCK="${WP_KG_WARPS_BLOCK}"
  -DKEM_PACK_TPB="${KEM_PACK_TPB}"
)

if [[ -n "${KEM_ENCAPS_LAUNCH_BOUNDS}" ]]; then
  COMMON_FLAGS+=(-DKEM_ENCAPS_LAUNCH_BOUNDS="${KEM_ENCAPS_LAUNCH_BOUNDS}")
fi

if [[ -n "${KEM_DECAPS_LAUNCH_BOUNDS}" ]]; then
  COMMON_FLAGS+=(-DKEM_DECAPS_LAUNCH_BOUNDS="${KEM_DECAPS_LAUNCH_BOUNDS}")
fi

if [[ -n "${ROCM_WAVE32_FLAG}" ]]; then
  COMMON_FLAGS+=("${ROCM_WAVE32_FLAG}")
fi

if [[ -n "${EXTRA_HIPCC_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_FLAGS_ARRAY=(${EXTRA_HIPCC_FLAGS})
  COMMON_FLAGS+=("${EXTRA_FLAGS_ARRAY[@]}")
fi

declare -a TARGETS=(
  "kyber512:1:2"
  "kyber768:1:3"
  "kyber1024:1:4"
  "aigisenc1:2:1"
  "aigisenc2:2:2"
  "aigisenc3:2:3"
  "aigisenc4:2:4"
)

FILTER=${1:-}

if ! command -v "${HIPCC}" >/dev/null 2>&1; then
  echo "[错误] 未找到 hipcc，请先安装 ROCm 并把 hipcc 加入 PATH"
  exit 1
fi

mkdir -p amd_results/build

for spec in "${TARGETS[@]}"; do
  IFS=':' read -r name alg mode <<<"${spec}"
  if [[ -n "${FILTER}" && "${name}" != "${FILTER}" ]]; then
    continue
  fi

  out="${name}_amd"
  echo "[build] ${out} (ALGORITHM=${alg} PARAM_MODE=${mode}, arch=${ROCM_ARCH}, opt=${OPT_FLAGS[*]}, KEM_TPB=${KEM_KEYGEN_TPB}/${KEM_ENCAPS_TPB}/${KEM_DECAPS_TPB}, bounds=${KEM_KEYPAIR_LAUNCH_BOUNDS}/${KEM_ENCAPS_LAUNCH_BOUNDS:-default}/${KEM_DECAPS_LAUNCH_BOUNDS:-default}, wpkg=${WP_KG_WARPS_BLOCK}, pack=${KEM_PACK_TPB})"
  "${HIPCC}" "${COMMON_FLAGS[@]}" \
    -DALGORITHM="${alg}" -DPARAM_MODE="${mode}" \
    -o "${out}" main.cu \
    2>&1 | tee "amd_results/build/${out}.log"
done

echo
echo "HIP 构建完成"
