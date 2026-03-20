#!/bin/sh
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

# run_makefile_tests.sh - Run tests for Makefile-built examples
#
# Finds all executables built by the Makefile build system, runs each one
# with a timeout, and reports pass/fail results.
#
# Usage:
#   ./run_makefile_tests.sh [--skip-file=FILE] [--timeout=SECONDS]
#
# The skip file should contain one test name per line (the EXAMPLE name).
# This is the same format produced by generate_skip_tests.py.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_FILE=""
TIMEOUT=120

for arg in "$@"; do
    case "$arg" in
        --skip-file=*)
            SKIP_FILE="${arg#--skip-file=}"
            ;;
        --timeout=*)
            TIMEOUT="${arg#--timeout=}"
            ;;
        --help|-h)
            echo "Usage: $0 [--skip-file=FILE] [--timeout=SECONDS]"
            echo ""
            echo "  --skip-file=FILE   File with test names to skip (one per line)."
            echo "                     Same format as generate_skip_tests.py output."
            echo "  --timeout=SECONDS  Per-test timeout (default: 120)"
            exit 0
            ;;
    esac
done

# Load skip list into a string for matching
SKIP_LIST=""
if [ -n "${SKIP_FILE}" ] && [ -f "${SKIP_FILE}" ]; then
    SKIP_LIST=$(cat "${SKIP_FILE}")
fi

is_skipped() {
    test_name="$1"
    echo "${SKIP_LIST}" | grep -qx "${test_name}" 2>/dev/null
}

TESTS_LIST="/tmp/makefile_tests_list_$$.txt"

# Find all Makefiles with EXAMPLE definitions, extract name, directory, and test args
find "${SCRIPT_DIR}" -name Makefile -path '*/Makefile' | sort | while IFS= read -r makefile; do
    dir=$(dirname "$makefile")
    example_name=$(grep '^EXAMPLE := ' "$makefile" 2>/dev/null | head -1 | sed 's/^EXAMPLE := //')

    # Skip Makefiles without EXAMPLE (parent Makefiles)
    [ -z "${example_name}" ] && continue

    # Skip if executable wasn't built
    [ ! -x "${dir}/${example_name}" ] && continue

    # Extract TEST_ARGS if defined (e.g. TEST_ARGS := graph4096.txt)
    test_args=$(grep '^TEST_ARGS := ' "$makefile" 2>/dev/null | head -1 | sed 's/^TEST_ARGS := //')

    echo "${dir}|${example_name}|${test_args}"
done > "${TESTS_LIST}"

# Count total tests upfront
NUM_TESTS=$(wc -l < "${TESTS_LIST}")

echo ""
echo "Found ${NUM_TESTS} Makefile-built test(s)"
if [ -n "${SKIP_FILE}" ] && [ -f "${SKIP_FILE}" ] && [ -s "${SKIP_FILE}" ]; then
    echo "Skip file: ${SKIP_FILE} ($(wc -l < "${SKIP_FILE}") entries)"
fi
echo ""

PASSED=0
FAILED=0
SKIPPED=0
CURRENT=0
FAILED_TESTS=""

# Run tests
while IFS='|' read -r dir example_name test_args; do
    CURRENT=$((CURRENT + 1))

    if is_skipped "${example_name}"; then
        SKIPPED=$((SKIPPED + 1))
        printf "(%d/%d) SKIP    %s\n" "${CURRENT}" "${NUM_TESTS}" "${example_name}"
        continue
    fi

    # Run the test from its own directory (so data files are found)
    printf "(%d/%d) RUN     %s\n" "${CURRENT}" "${NUM_TESTS}" "${example_name}"
    if (cd "${dir}" && timeout "${TIMEOUT}" "./${example_name}" ${test_args}) > "/tmp/test_${example_name}.log" 2>&1; then
        PASSED=$((PASSED + 1))
        printf "(%d/%d) PASS    %s\n" "${CURRENT}" "${NUM_TESTS}" "${example_name}"
    else
        exit_code=$?
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS} ${example_name}"
        if [ "${exit_code}" -eq 124 ]; then
            printf "(%d/%d) TIMEOUT %s (after %ss)\n" "${CURRENT}" "${NUM_TESTS}" "${example_name}" "${TIMEOUT}"
        else
            printf "(%d/%d) FAIL    %s (exit code %d)\n" "${CURRENT}" "${NUM_TESTS}" "${example_name}" "${exit_code}"
        fi
        # Print last 20 lines of output for failed tests
        echo "--- output (last 20 lines) ---"
        tail -20 "/tmp/test_${example_name}.log"
        echo "--- end output ---"
    fi
done < "${TESTS_LIST}"

rm -f "${TESTS_LIST}"

echo ""
echo "=========================================="
RAN=$((PASSED + FAILED))
echo "Makefile test results: ${RAN} tests ran, ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped (${NUM_TESTS} total)"
echo "=========================================="

if [ -n "${FAILED_TESTS}" ]; then
    echo ""
    echo "Failed tests:${FAILED_TESTS}"
    exit 1
fi
