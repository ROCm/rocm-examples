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
echo "Profiling workload; filtering for kernel substring vgprbound"
echo "==============================================================================="
# Kernels are specified as a substring list. The following matches all kernel names which contain "vgprbound". Roofline
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
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_first --block 7 --no-roof -- $EXAMPLE_WORKLOAD
$EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_second --block 7 --no-roof -- $EXAMPLE_WORKLOAD
$EXAMPLE_TOOL analyze --path workloads/${EXAMPLE_BIN}_first/* --path workloads/${EXAMPLE_BIN}_second/* --block 7

# Filtering for metric sets is only supported in rocprof-compute 3.3.0 and later (ROCm 7.1 and later). We verify the
# version before proceeding to the next example.
RPCVER=$($EXAMPLE_TOOL --version | grep version | awk '{print $3}')
if ! printf '3.3.0\n%s\n' $RPCVER | sort -V -C; then
    echo "rocprof-compute only supports sets profiling starting from v3.3.0. Skipping sets example."
else
    echo "==============================================================================="
    echo "Profiling workload; filtering for metrics set: Wavefront Launch Statistics"
    echo "==============================================================================="
    # A metric set contains a subset of metrics that can be collected in a single pass. 
    # This is useful for minimizing profiling overhead by collecting only the counters 
    # of interest. Note that rocprof-compute might collect other metrics as well, but 
    # only the metrics that are part of the set will be meaningful.
    # 
    # To list available sets, use the '--list-sets' flag:
    # rocprof-compute profile --list-sets
    #
    # After obtaining the list of sets, specify a set using the '--set <set_name>' flag. Note that this flag cannot be
    # used together with the '--roof-only' and '--block' flags.
    $EXAMPLE_TOOL profile --name ${EXAMPLE_BIN}_launch --set launch_stats -- $EXAMPLE_WORKLOAD
    # In this example, block 7.1 is analyzed because the 'launch_stats' set collects sub-blocks 7.1.0 through 7.1.8.
    $EXAMPLE_TOOL analyze --path workloads/${EXAMPLE_BIN}_launch/* --block 7.1
fi
exit 0
