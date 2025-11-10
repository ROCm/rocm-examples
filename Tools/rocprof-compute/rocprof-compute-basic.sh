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
echo "Profiling workload"
echo "==============================================================================="
$EXAMPLE_TOOL profile --name $EXAMPLE_BIN -- $EXAMPLE_WORKLOAD

# Roofline profiling is part of the default profiling set. If you want to perform roofline analysis only, uncommment the
# following section.
#echo "==============================================================================="
#echo "Profiling workload for roofline analysis"
#echo "==============================================================================="
# The following will only collect metrics which are required for roofline analysis.
#$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_roofline --roof-only -- $EXAMPLE_WORKLOAD

# The rocpd file format is only supported for rocprof-compute >= 3.3.0 (= at least ROCm 7.1). We perform a version check
# before continuing to the next example
RPCVER=$($EXAMPLE_TOOL --version | grep version | awk '{print $3}')
if ! printf '3.3.0\n%s\n' $RPCVER | sort -V -C; then
    echo "rocprof-compute only supports the rocpd format starting from v3.3.0. Skipping rocpd example."
else
    echo "==============================================================================="
    echo "Profiling workload and saving in rocpd file format"
    echo "==============================================================================="
    # The following will store the results in *.rocpd file(s). rocprof-compute also supports csv (default).
    ${EXAMPLE_TOOL} profile \
        --name ${EXAMPLE_BIN}_rocpd \
        --format-rocprof-output rocpd \
        --retain-rocpd-output \
        --no-roof \
        -- $EXAMPLE_WORKLOAD
fi

# Check if the workload directory exists before analyzing
if [ ! -d "workloads/$EXAMPLE_BIN" ]; then
    echo "Error: workloads/$EXAMPLE_BIN directory not found" >&2
    exit 1
fi

echo "==============================================================================="
echo "Performing CLI analysis: System Speed-of-Light"
echo "==============================================================================="
# Block 2: Speed-of-Light
$EXAMPLE_TOOL analyze --path workloads/$EXAMPLE_BIN/* --block 2


echo "==============================================================================="
echo "Performing CLI analysis: Memory chart"
echo "==============================================================================="
# Block 3: Memory chart
$EXAMPLE_TOOL analyze --path workloads/$EXAMPLE_BIN/* --block 3

echo "==============================================================================="
echo "Performing CLI analysis: Roofline"
echo "==============================================================================="
# Block 4: Roofline
$EXAMPLE_TOOL analyze --path workloads/$EXAMPLE_BIN/* --block 4

exit 0
