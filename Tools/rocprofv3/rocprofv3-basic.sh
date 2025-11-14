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

EXAMPLE_TOOL="rocprofv3"
EXAMPLE_BIN="${EXAMPLE_TOOL}_matmul"
EXAMPLE_WORKLOAD="./$EXAMPLE_BIN"

# Check for existence of tool
if ! [ -x "$(command -v $EXAMPLE_TOOL)" ]; then
    echo "Error: Could not find $EXAMPLE_TOOL in the PATH." >&2
    exit 1
fi

if [ ! -f $EXAMPLE_BIN ]; then
    echo "Error: Could not find $EXAMPLE_BIN in working directory." >&2
    exit 1
fi

# Exit on any error
set -e

echo "==============================================================================="
echo "Runtime trace (rocpd format)"
echo "==============================================================================="
# Target most relevant tracing options but exclude low-level APIs such as HSA or HIP compiler API.
# By default, rocprofv3 saves its output in a new directory: %hostname%/%pid%. Explicitly setting the output directory
# will save the results in $OUTDIR/%hostname%/%pid%.
OUTDIR="${EXAMPLE_TOOL}-runtime-trace-rocpd" \
    $EXAMPLE_TOOL \
    --runtime-trace \
    --output-directory %env{OUTDIR}% \
    -- $EXAMPLE_WORKLOAD

echo "==============================================================================="
echo "Runtime trace (Perfetto format)"
echo "==============================================================================="
# Same as above, but the output can be analyzed with Perfetto.
OUTDIR="${EXAMPLE_TOOL}-runtime-trace-perfetto" \
    $EXAMPLE_TOOL \
    --runtime-trace \
    --output-directory %env{OUTDIR}% \
    --output-format pftrace \
    -- $EXAMPLE_WORKLOAD

echo "==============================================================================="
echo "System trace (CSV format)"
echo "==============================================================================="
# A system trace is an all-inclusive option which also includes low-level APIs.
OUTDIR="${EXAMPLE_TOOL}-system-trace" \
    $EXAMPLE_TOOL \
    --sys-trace \
    --output-directory %env{OUTDIR}% \
    --output-format csv \
    -- $EXAMPLE_WORKLOAD

exit 0
