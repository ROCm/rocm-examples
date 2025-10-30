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

EXAMPLE_TOOL := rocprof-compute
EXAMPLE_WORKLOAD := "${EXAMPLE_TOOL}_vcopy"
USED_GPU :=

# Check for existence of tool
if ! [ -x "$(command -v ${EXAMPLE_TOOL})" ]; then
    echo "Error: Could not find ${EXAMPLE_TOOL} in the PATH." >&2
    exit 1
fi

echo "==============================================================================="
echo "Profiling workload; filtering for kernel substring vecCopy"
echo "==============================================================================="
# Kernels are specified as a substring list. The following matches all kernel names which contain "vecCopy".
WORKDIR := "${EXAMPLE_WORKLOAD}_substr"
${EXAMPLE_TOOL} profile --name ${WORKDIR} --kernel vecCopy -- ./${EXAMPLE_WORKLOAD} -n 1048576 -b 256
if [ $? -eq 1 ]; then
    echo "${EXAMPLE_TOOL} returned an error." >&2
    exit 1
fi

echo "==============================================================================="
echo "Profiling workload; filtering for Wavefront Launch Statistics"
echo "==============================================================================="
# It is possible to only collect some metrics. The list of supported hardware report blocks can be obtained with
# rocprof-compute profile --list-metrics
WORKDIR := "${EXAMPLE_WORKLOAD}_wavefront"
${EXAMPLE_TOOL} profile --name ${WORKDIR} --block 7 -- ./${EXAMPLE_WORKLOAD} -n 1048576 -b 256
if [ $? -eq 1 ]; then
    echo "${EXAMPLE_TOOL} returned an error." >&2
    exit 1
fi

echo "==============================================================================="
echo "Profiling two runs for comparative analysis"
echo "==============================================================================="
${EXAMPLE_TOOL} profile --name ${EXAMPLE_WORKLOAD}_first -- ./${EXAMPLE_WORKLOAD} -n 1048576 -b 256
if [ $? -eq 1 ]; then
    echo "${EXAMPLE_TOOL} returned an error." >&2
    exit 1
fi
${EXAMPLE_TOOL} profile --name ${EXAMPLE_WORKLOAD}_second -- ./${EXAMPLE_WORKLOAD} -n 1048576 -b 256
if [ $? -eq 1 ]; then
    echo "${EXAMPLE_TOOL} returned an error." >&2
    exit 1
fi
${EXAMPLE_TOOL} analyze --path workload/${EXAMPLE_WORKLOAD}_first/* --path workload/${EXAMPLE_WORKLOAD}_second
if [ $? -eq 1 ]; then
    echo "${EXAMPLE_TOOL} returned an error." >&2
    exit 1
fi

echo "==============================================================================="
echo "Profiling with PC sampling"
echo "==============================================================================="
# At the moment, block 21 (the block containing PC metrics) must be explicitly enabled.
${EXAMPLE_TOOL} profile \
    --name ${EXAMPLE_WORKLOAD}_pc \
    --block 21 \
    --pc-sampling-method stochastic \
    -- ./${EXAMPLE_WORKLOAD} -n 1048576 -b 256
if [ $? -eq 1 ]; then
    echo "${EXAMPLE_TOOL} returned an error." >&2
    exit 1
fi

exit 0
