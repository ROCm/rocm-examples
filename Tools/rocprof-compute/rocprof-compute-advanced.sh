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
EXAMPLE_BIN="${EXAMPLE_TOOL}-occupancy"
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
echo "Profiling workload; filtering for kernel substring vecCopy"
echo "==============================================================================="
# Kernels are specified as a substring list. The following matches all kernel names which contain "vecCopy". Roofline
# profiling is disabled to save profiling time.
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_substr --kernel vgprbound --no-roof -- $EXAMPLE_WORKLOAD
# Notice the "top kernels" only shows "vgprbound" kernels, compared to the full profile capture  
$EXAMPLE_TOOL analyze --path workloads/${EXAMPLE_BIN}_substr/* --block 7 

echo "==============================================================================="
echo "Profiling workload; filtering for System Speed-of-Light profiling and analysis"
echo "==============================================================================="
# It is possible to only collect some metrics. The list of supported hardware report blocks can be obtained with
# rocprof-compute profile --list-metrics
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_sol --block 2 -- $EXAMPLE_WORKLOAD
$EXAMPLE_TOOL analyze --path workloads/${EXAMPLE_BIN}_sol/* --block 2  


echo "==============================================================================="
echo "Profiling two runs for comparative analysis"
echo "==============================================================================="
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_first --block 2 --no-roof -- $EXAMPLE_WORKLOAD
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_second --block 2 --no-roof -- $EXAMPLE_WORKLOAD
$EXAMPLE_TOOL analyze --path workloads/${EXAMPLE_BIN}_first/* --path workloads/${EXAMPLE_BIN}_second/*

exit 0
