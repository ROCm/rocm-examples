#!/usr/bin/env bash
set -euo pipefail

NVCC="${NVCC:-nvcc}"
SRC="main.cu"
FILTER="${1:-}"

# RTX 4090 = Ada Lovelace / sm_89. Override with:
#   CUDA_ARCH=sm_86 ./build_all.sh
CUDA_ARCH="${CUDA_ARCH:-sm_89}"
KEM_SERIAL_TPB="${KEM_SERIAL_TPB:-64}"
FLAGS=(-O3 -std=c++14 --expt-relaxed-constexpr -DKEM_SERIAL_TPB="${KEM_SERIAL_TPB}")
ARCH=(-arch="${CUDA_ARCH}")

if ! command -v "${NVCC}" >/dev/null 2>&1; then
  echo "[error] nvcc not found in PATH" >&2
  exit 1
fi

targets=(
  "kyber512 1 2"
  "kyber768 1 3"
  "kyber1024 1 4"
  "aigisenc1 2 1"
  "aigisenc2 2 2"
  "aigisenc3 2 3"
  "aigisenc4 2 4"
)

for entry in "${targets[@]}"; do
  read -r name alg mode <<<"${entry}"
  if [[ -n "${FILTER}" && "${FILTER}" != "${name}" ]]; then
    continue
  fi

  echo "[build] ${name} (ALGORITHM=${alg} PARAM_MODE=${mode} ARCH=${CUDA_ARCH} KEM_SERIAL_TPB=${KEM_SERIAL_TPB})"
  "${NVCC}" "${FLAGS[@]}" "${ARCH[@]}" \
    -DALGORITHM="${alg}" -DPARAM_MODE="${mode}" \
    -o "${name}" "${SRC}"
  echo "[ok] ${name}"
done

echo
echo "build complete"
