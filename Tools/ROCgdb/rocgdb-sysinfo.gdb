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
echo Showing system information\n
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
echo Agent information\n
echo -------------------------------------------------------------------------------\n
info agents

echo -------------------------------------------------------------------------------\n
echo Queue information\n
echo -------------------------------------------------------------------------------\n
info queues

echo -------------------------------------------------------------------------------\n
echo Kernel dispatch information\n
echo -------------------------------------------------------------------------------\n
info dispatches

echo -------------------------------------------------------------------------------\n
echo CPU/GPU thread information\n
echo -------------------------------------------------------------------------------\n
# Threads can be named. The following command will set the current thread's name which will then appear in the output
# afterwards. 'thread find <regex>' (not shown here) can be used to find a specific thread.
thread name rocgdb-example-thread
info threads

echo -------------------------------------------------------------------------------\n
echo Lane information\n
echo -------------------------------------------------------------------------------\n
info lanes

continue
