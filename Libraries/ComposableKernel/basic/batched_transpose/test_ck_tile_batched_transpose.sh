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

EXE=$1

for pr in "fp8" "fp16" "bf16"; do
for pipeline in "0" "1"; do
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=32 -H=1 -W=32 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=64 -H=1 -W=64 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=2 -C=12 -H=1 -W=32 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=3 -C=1334 -H=1 -W=37 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=4 -C=27 -H=1 -W=32 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=5 -C=1234 -H=1 -W=12 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=1 -H=1 -W=1 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=1 -H=1 -W=1 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=128 -C=1024 -H=64 -W=64 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=128 -C=1024 -H=64 -W=64 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=16 -C=64 -H=32 -W=128 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=16 -C=64 -H=128 -W=32 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=2048 -H=1 -W=1 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=2048 -H=1 -W=1 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=1 -H=1024 -W=1024 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=1 -H=1024 -W=1024 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=8 -C=16 -H=8 -W=16 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=8 -C=16 -H=8 -W=16 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=64 -H=1 -W=1024 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=64 -H=1024 -W=1 -layout_in='NHWC' -layout_out='NCHW'

done
done
