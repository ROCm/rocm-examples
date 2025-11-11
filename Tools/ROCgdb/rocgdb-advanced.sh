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

EXAMPLE_TOOL="rocgdb"
EXAMPLE_BIN="rocgdb-matmul"
EXAMPLE_WORKLOAD="./$EXAMPLE_BIN"

# Check for existence of tool
if ! [ -x "$(command -v $EXAMPLE_TOOL)" ]; then
    echo "Error: Could not find $EXAMPLE_TOOL in the PATH." >&2
    exit 1
fi

# Exit on any error
set -e

# ROCgdb is a fork of the standard gdb. We will not show how to use basic gdb commands here (consult the (ROC)gdb manual
# on how to use plain gdb) and instead focus on ROCgdb's additional features.

# We are running multiple examples in batch mode, i.e. non-interactive. The necessary ROCgdb commands are loaded from
# gdb script files; they can be used in interactive mode in the same way as shown in the scripts. ROCgdb's output can
# be inspected by opening the generated log files.

# GPU data examination
$EXAMPLE_TOOL --batch --command=rocgdb-gpu-data.gdb $EXAMPLE_WORKLOAD > rocgdb-gpu-data.log

# Modifying wavefront execution
$EXAMPLE_TOOL --batch --command=rocgdb-wave-exec.gdb $EXAMPLE_WORKLOAD > rocgdb-wave-exec.log
