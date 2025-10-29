#!/bin/bash
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
EXE=$1
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=2 -warmup=0 -repeat=1'

run_tests() {
    for m in 128 1024; do
        for n in 128 2048; do
            for k in 128 4096; do

                $EXE -m=$m -n=$n -k=$k -stride_a=0 -stride_b=0 -stride_c=0 -prec=$1 $COMMON_ARGS
                if [ $? -eq 0 ]; then
                    echo "Success: Test with m=$m, n=$n, k=$k executed successfully."
                else
                    echo "Error: Test with m=$m, n=$n, k=$k failed to execute properly."
                    # Optionally, exit or break if you need to halt further execution
                    # exit 1
                fi

            done
        done
    done
}

set -x

run_tests "bf16"
run_tests "fp16"

set +x
