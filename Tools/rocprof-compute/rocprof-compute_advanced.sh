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

EXAMPLE_TOOL="rocprof-compute"
EXAMPLE_BIN="${EXAMPLE_TOOL}_vcopy"
EXAMPLE_WORKLOAD="./$EXAMPLE_BIN -n 1048576 -b 256"

# Check for existence of tool
if ! [ -x "$(command -v $EXAMPLE_TOOL)" ]; then
    echo "Error: Could not find $EXAMPLE_TOOL in the PATH." >&2
    exit 1
fi

# Exit on any error
set -e

echo "==============================================================================="
echo "Profiling workload; filtering for kernel substring vecCopy"
echo "==============================================================================="
# Kernels are specified as a substring list. The following matches all kernel names which contain "vecCopy". Roofline
# profiling is disabled to save profiling time.
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_substr --kernel vecCopy --no-roof -- $EXAMPLE_WORKLOAD

echo "==============================================================================="
echo "Profiling workload; filtering for Wavefront Launch Statistics"
echo "==============================================================================="
# It is possible to only collect some metrics. The list of supported hardware report blocks can be obtained with
# rocprof-compute profile --list-metrics
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_wavefront --block 7 -- $EXAMPLE_WORKLOAD

echo "==============================================================================="
echo "Profiling two runs for comparative analysis"
echo "==============================================================================="
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_first --no-roof -- $EXAMPLE_WORKLOAD
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_second --no-roof -- $EXAMPLE_WORKLOAD
$EXAMPLE_TOOL analyze --path workloads/${EXAMPLE_BIN}_first/* --path workloads/${EXAMPLE_BIN}_second/*

echo "==============================================================================="
echo "Profiling with PC sampling"
echo "==============================================================================="
# At the moment, block 21 (the block containing PC metrics) must be explicitly enabled.
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_pc --block 21 --pc-sampling-method stochastic -- $EXAMPLE_WORKLOAD

exit 0
