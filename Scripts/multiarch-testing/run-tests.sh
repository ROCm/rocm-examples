#!/usr/bin/env bash
# run-tests.sh — Build the Docker image (if needed) and run the rocm-examples
# ctest suite inside a container with GPU access.
#
# Usage:
#   ./run-tests.sh [/path/to/rocm-examples] [ctest-filter]
#
# Examples:
#   ./run-tests.sh                            # run all tests
#   ./run-tests.sh ~/rocm-examples            # explicit source path
#   ./run-tests.sh ~/rocm-examples 'hip_.*'  # run only HIP tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="rocm-examples-test"
EXAMPLES_DIR="${1:-${SCRIPT_DIR}/../../../rocm-examples}"
CTEST_FILTER="${2:-}"

if [[ ! -d "${EXAMPLES_DIR}" ]]; then
    echo "ERROR: rocm-examples directory not found: ${EXAMPLES_DIR}" >&2
    echo "Usage: $0 [/path/to/rocm-examples] [ctest-filter]" >&2
    exit 1
fi

EXAMPLES_DIR="$(realpath "${EXAMPLES_DIR}")"

# ── Detect render group GID from the host ─────────────────────────────────────
RENDER_GID="$(getent group render 2>/dev/null | cut -d: -f3 || echo 109)"

# ── Build the image ───────────────────────────────────────────────────────────
echo "==> Building image '${IMAGE_NAME}' (RENDER_GID=${RENDER_GID})..."
docker build \
    --build-arg "RENDER_GID=${RENDER_GID}" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

# ── Assemble ctest args ───────────────────────────────────────────────────────
CTEST_ARGS="--output-on-failure"
if [[ -n "${CTEST_FILTER}" ]]; then
    CTEST_ARGS="${CTEST_ARGS} -R ${CTEST_FILTER}"
fi

# ── Run tests inside the container ───────────────────────────────────────────
echo "==> Running tests from: ${EXAMPLES_DIR}"
echo "==> ctest args: ${CTEST_ARGS}"
echo ""

docker run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    -e "CTEST_ARGS=${CTEST_ARGS}" \
    -v "${EXAMPLES_DIR}:/workspace/rocm-examples:ro" \
    "${IMAGE_NAME}" \
    bash -c "
        set -euo pipefail

        echo '==> CMake configure...'
        cmake -B /workspace/build \
              -S /workspace/rocm-examples \
              -DROCM_PATH=\${ROCM_PATH} \
              -DROCM_EXAMPLES_ENABLE_ROCDECODE=OFF \
              -Wno-dev \
              -G Ninja

        echo '==> Build...'
        cmake --build /workspace/build -j\$(nproc)

        echo '==> Running tests...'
        cd /workspace/build
        ctest -j1 \${CTEST_ARGS}
    "
