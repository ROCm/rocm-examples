#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# run-tests.sh — Run the rocm-examples ctest suite inside a distro container.
#
# Usage:
#   ./run-tests.sh [OPTIONS] [/path/to/rocm-examples] [ctest-filter]
#
# Options:
#   --distro DISTRO         Target distro (default: ubuntu-24.04)
#                           One of: ubuntu-24.04, ubuntu-26.04, rhel-10.1,
#                                   oracle-10, rocky-9
#   --rebuild               Force Docker image rebuild even if it already exists
#   --rebuild-workspace     Wipe and rebuild the CMake workspace before testing
#
# Positional args (after any --flags):
#   /path/to/rocm-examples  Source directory (default: auto-detected from script location)
#   ctest-filter            Regex passed to ctest -R (default: run all tests)
#
# Examples:
#   ./run-tests.sh                                         # ubuntu-24.04, all tests
#   ./run-tests.sh --distro rhel-10.1                     # test on RHEL 10.1
#   ./run-tests.sh --rebuild                              # force image rebuild
#   ./run-tests.sh --rebuild-workspace                    # wipe build, reconfigure
#   ./run-tests.sh ~/rocm-examples 'rocblas_.*'           # filter tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
DISTRO="ubuntu-24.04"
REBUILD_IMAGE=false
REBUILD_WORKSPACE=false
EXAMPLES_DIR=""
CTEST_FILTER=""

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --distro)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --distro requires an argument" >&2; exit 1
            fi
            DISTRO="$2"; shift 2 ;;
        --rebuild)
            REBUILD_IMAGE=true; shift ;;
        --rebuild-workspace)
            REBUILD_WORKSPACE=true; shift ;;
        --*)
            echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "${EXAMPLES_DIR}" ]]; then
                EXAMPLES_DIR="$1"
            elif [[ -z "${CTEST_FILTER}" ]]; then
                CTEST_FILTER="$1"
            else
                echo "ERROR: Unexpected argument: $1" >&2; exit 1
            fi
            shift ;;
    esac
done

# ── Resolve paths ─────────────────────────────────────────────────────────────
EXAMPLES_DIR="${EXAMPLES_DIR:-${SCRIPT_DIR}/../../../rocm-examples}"

if [[ ! -d "${EXAMPLES_DIR}" ]]; then
    echo "ERROR: rocm-examples directory not found: ${EXAMPLES_DIR}" >&2
    echo "Usage: $0 [OPTIONS] [/path/to/rocm-examples] [ctest-filter]" >&2
    exit 1
fi
EXAMPLES_DIR="$(realpath "${EXAMPLES_DIR}")"
BUILD_DIR="${EXAMPLES_DIR}/build-multiarch-${DISTRO}"

# ── Validate distro ───────────────────────────────────────────────────────────
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.${DISTRO}"
if [[ ! -f "${DOCKERFILE}" ]]; then
    VALID=()
    for f in "${SCRIPT_DIR}"/Dockerfile.*; do
        [[ -f "$f" ]] && VALID+=("${f##*Dockerfile.}")
    done
    echo "ERROR: Unknown distro '${DISTRO}'. Valid options: ${VALID[*]}" >&2
    exit 1
fi

IMAGE_NAME="rocm-examples-test-${DISTRO}"

# ── Detect render group GID from the host ─────────────────────────────────────
RENDER_GID="$(getent group render 2>/dev/null | cut -d: -f3 || echo 109)"

# ── Build the image (if needed) ───────────────────────────────────────────────
if [[ "${REBUILD_IMAGE}" == true ]] || ! docker image inspect "${IMAGE_NAME}" &>/dev/null; then
    echo "==> Building image '${IMAGE_NAME}' (RENDER_GID=${RENDER_GID})..."
    docker build \
        --build-arg "RENDER_GID=${RENDER_GID}" \
        -t "${IMAGE_NAME}" \
        -f "${DOCKERFILE}" \
        "${SCRIPT_DIR}"
else
    echo "==> Image '${IMAGE_NAME}' already exists, skipping build (use --rebuild to force)."
fi

# ── Persistent build directory ────────────────────────────────────────────────
# Build artifacts are kept between runs so ctest can re-run without rebuilding.
if [[ "${REBUILD_WORKSPACE}" == true ]]; then
    echo "==> Wiping build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi
# Sticky bit + world-writable so the container's non-root user (UID 1000) can write.
mkdir -p "${BUILD_DIR}"
chmod 1777 "${BUILD_DIR}"

# ── Run tests inside the container ───────────────────────────────────────────
echo "==> Running tests from:  ${EXAMPLES_DIR}"
echo "==> Distro:              ${DISTRO}"
echo "==> Build dir (host):    ${BUILD_DIR}"
echo "==> ctest filter:        ${CTEST_FILTER:-<all>}"
echo ""

docker run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    -v "${EXAMPLES_DIR}:/workspace/rocm-examples:ro" \
    -v "${BUILD_DIR}:/workspace/build" \
    -e "CTEST_FILTER=${CTEST_FILTER}" \
    "${IMAGE_NAME}" \
    bash -c '
        set -euo pipefail

        if [[ ! -f /workspace/build/CMakeCache.txt ]]; then
            echo "==> CMake configure..."
            cmake -B /workspace/build \
                  -S /workspace/rocm-examples \
                  -DROCM_PATH="${ROCM_PATH}" \
                  -DROCM_EXAMPLES_ENABLE_ROCDECODE=OFF \
                  -Wno-dev \
                  -G Ninja

            echo "==> Build..."
            cmake --build /workspace/build -j$(nproc)
        else
            echo "==> Existing build found, skipping configure+build (use --rebuild-workspace to wipe)."
        fi

        echo "==> Running tests..."
        cd /workspace/build
        CTEST_CMD=(ctest -j1 --output-on-failure)
        if [[ -n "${CTEST_FILTER:-}" ]]; then
            CTEST_CMD+=(-R "${CTEST_FILTER}")
        fi
        "${CTEST_CMD[@]}"
    '
