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
EXAMPLE_INSTRUMENTER="rocprof-sys-instrument"
EXAMPLE_SAMPLER="rocprof-sys-sample"

EXAMPLE_BIN="${EXAMPLE}_matmul"
EXAMPLE_WORKLOAD="./$EXAMPLE_BIN --A_rows 8192 --A_cols 4096 --B_cols 4096"

# Check for existence of tools
if ! [ -x "$(command -v $EXAMPLE_INSTRUMENTER)" ]; then
    echo "Error: Could not find $EXAMPLE_INSTRUMENTER in the PATH." >&2
    exit 1
fi
if ! [ -x "$(command -v $EXAMPLE_SAMPLER)" ]; then
    echo "Error: Could not find $EXAMPLE_SAMPLER in the PATH." >&2
    exit 1
fi

# Exit on any error
set -e

echo "==============================================================================="
echo "Basic call-stack sampling"
echo "==============================================================================="
# Without any additional arguments, rocprof-sys-sample will perform timer-based sampling per thread and no process-wide
# sampling.
$EXAMPLE_SAMPLER -- $EXAMPLE_WORKLOAD

echo "==============================================================================="
echo "Basic profiling and tracing"
echo "==============================================================================="
# By default, the profiling results are dumped to stdout. Here we save them as JSON.
# The tracing results can be analyzed with Perfetto.
$EXAMPLE_SAMPLER --profile --profile-format json --trace -- $EXAMPLE_WORKLOAD

echo "==============================================================================="
echo "Basic runtime instrumentation"
echo "==============================================================================="
# Instrument with default settings. These are:
# * Skip dynamic callsites (e.g. function pointers)
# * Only functions with at least 1024 instructions are instrumented
# * Skip instrumentation points which require traps
# * Skip instrumenting loops within a function body
# * Skip instrumenting functions with overlapping bodies and single functions with multiple entry points
$EXAMPLE_INSTRUMENTER -- $EXAMPLE_WORKLOAD

exit 0
