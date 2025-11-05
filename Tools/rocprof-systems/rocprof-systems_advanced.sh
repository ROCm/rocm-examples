#!/bin/bash
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

EXAMPLE="rocprof-systems"
EXAMPLE_QUERY="rocprof-sys-avail"
EXAMPLE_INSTRUMENTER="rocprof-sys-instrument"
EXAMPLE_SAMPLER="rocprof-sys-sample"
EXAMPLE_RUNNER="rocprof-sys-run"

EXAMPLE_BIN="${EXAMPLE}_matmul"
EXAMPLE_BIN_USERAPI="${EXAMPLE_BIN}_userapi"

# Check for existence of tools
if ! [ -x "$(command -v $EXAMPLE_QUERY)" ]; then
    echo "Error: Could not find $EXAMPLE_QUERY in the PATH." >&2
    exit 1
fi
if ! [ -x "$(command -v $EXAMPLE_INSTRUMENTER)" ]; then
    echo "Error: Could not find $EXAMPLE_INSTRUMENTER in the PATH." >&2
    exit 1
fi
if ! [ -x "$(command -v $EXAMPLE_SAMPLER)" ]; then
    echo "Error: Could not find $EXAMPLE_SAMPLER in the PATH." >&2
    exit 1
fi
if ! [ -x "$(command -v $EXAMPLE_RUNNER)" ]; then
    echo "Error: Could not find $EXAMPLE_RUNNER in the PATH." >&2
    exit 1
fi

# Exit on any error
set -e

echo "==============================================================================="
echo "Instrumenting HIP API calls"
echo "==============================================================================="
# Binary-rewrite our executable to include instrumentation points
$EXAMPLE_INSTRUMENTER --output ${EXAMPLE_BIN}.inst -- $EXAMPLE_BIN
# Restrict our experiment to HIP API calls, ignore everything else
$EXAMPLE_RUNNER --profile --trace --rocm-domains hip_api -- ./${EXAMPLE_BIN}.inst

echo "==============================================================================="
echo "Instrumenting CPU performance counters"
echo "==============================================================================="
# Instrument the application.
$EXAMPLE_INSTRUMENTER --output ${EXAMPLE_BIN}.papi.inst -- $EXAMPLE_BIN
# We are interested in the following CPU counters: total cycles, total instructions completed, L1-L3 total cache misses
$EXAMPLE_RUNNER \
    --profile \
    --cpu-events PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM \
    -- ./${EXAMPLE_BIN}.papi.inst

echo "==============================================================================="
echo "Instrumenting with user API"
echo "==============================================================================="
# The following commmands first perform a binary rewrite of the executable (for instrumentation) and then run a profile
# and trace of the application which includes user-defined instrumentation regions.
$EXAMPLE_INSTRUMENTER --min-instructions 512 -o ${EXAMPLE_BIN_USERAPI}.inst -- $EXAMPLE_BIN_USERAPI
$EXAMPLE_RUNNER --profile --trace -- ${EXAMPLE_BIN_USERAPI}.inst

exit 0
