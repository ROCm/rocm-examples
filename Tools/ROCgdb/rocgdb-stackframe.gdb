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

echo ===============================================================================\n
echo Examining stackframe\n
echo ===============================================================================\n

# By default, GPU code objects are not loaded until the first kernel is launched. Using a kernel name for setting a
# GPU breakpoint will mark the breakpoint as 'pending'. In batch mode, pending breakpoints must be enabled. In
# interactive mode, ROCgdb will ask the user if the unknown breakpoint is supposed to be pending.
set breakpoint pending on

# Temporary breakpoint in GPU kernel code - will be automatically deleted after the first encounter. Since this is a
# pending breakpoint (it is set before the GPU kernel code is loaded) ROCgdb will issue a warning:
# "Function "matrix_multiplication_kernel" not defined." In this case, it can be safely ignored.
tbreak matrix_multiplication_kernel

run

echo -------------------------------------------------------------------------------\n
echo Backtrace\n
echo -------------------------------------------------------------------------------\n
backtrace

echo -------------------------------------------------------------------------------\n
echo High-level frame information\n
echo -------------------------------------------------------------------------------\n
frame

echo -------------------------------------------------------------------------------\n
echo Detailed frame information\n
echo -------------------------------------------------------------------------------\n
info frame

echo -------------------------------------------------------------------------------\n
echo Frame argument information\n
echo -------------------------------------------------------------------------------\n
info args

echo -------------------------------------------------------------------------------\n
echo Local variable information\n
echo -------------------------------------------------------------------------------\n
info locals

continue
