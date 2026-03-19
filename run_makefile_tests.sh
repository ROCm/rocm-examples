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
#   ./run_makefile_tests.sh [--skip-file FILE] [--timeout SECONDS]
#
# The skip file should contain one test name per line (the EXAMPLE name).

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

PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0
FAILED_TESTS=""

# Find all Makefiles with EXAMPLE definitions, extract name and directory
find "${SCRIPT_DIR}" -name Makefile -path '*/Makefile' | sort | while IFS= read -r makefile; do
    dir=$(dirname "$makefile")
    example_name=$(grep '^EXAMPLE := ' "$makefile" 2>/dev/null | head -1 | sed 's/^EXAMPLE := //')

    # Skip Makefiles without EXAMPLE (parent Makefiles)
    [ -z "${example_name}" ] && continue

    # Skip if executable wasn't built
    [ ! -x "${dir}/${example_name}" ] && continue

    echo "${dir}|${example_name}"
done > /tmp/makefile_tests_list.txt

# Count and run
while IFS='|' read -r dir example_name; do
    TOTAL=$((TOTAL + 1))

    if is_skipped "${example_name}"; then
        SKIPPED=$((SKIPPED + 1))
        printf "  SKIP  %s\n" "${example_name}"
        continue
    fi

    # Run the test with timeout
    printf "  RUN   %s\n" "${example_name}"
    if timeout "${TIMEOUT}" "${dir}/${example_name}" > "/tmp/test_${example_name}.log" 2>&1; then
        PASSED=$((PASSED + 1))
        printf "  PASS  %s\n" "${example_name}"
    else
        exit_code=$?
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS} ${example_name}"
        if [ "${exit_code}" -eq 124 ]; then
            printf "  TIMEOUT %s (after %ss)\n" "${example_name}" "${TIMEOUT}"
        else
            printf "  FAIL  %s (exit code %d)\n" "${example_name}" "${exit_code}"
        fi
        # Print last 20 lines of output for failed tests
        echo "--- output (last 20 lines) ---"
        tail -20 "/tmp/test_${example_name}.log"
        echo "--- end output ---"
    fi
done < /tmp/makefile_tests_list.txt

rm -f /tmp/makefile_tests_list.txt

echo ""
echo "=========================================="
echo "Makefile test results:"
echo "  Total:   ${TOTAL}"
echo "  Passed:  ${PASSED}"
echo "  Failed:  ${FAILED}"
echo "  Skipped: ${SKIPPED}"
echo "=========================================="

if [ -n "${FAILED_TESTS}" ]; then
    echo ""
    echo "Failed tests:${FAILED_TESTS}"
    exit 1
fi
