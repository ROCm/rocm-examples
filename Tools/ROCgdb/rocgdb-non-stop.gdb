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
echo Debugging in non-stop mode\n
echo ===============================================================================\n

# By default, GPU code objects are not loaded until the first kernel is launched. Using a kernel name for setting a
# GPU breakpoint will mark the breakpoint as 'pending'. In batch mode, pending breakpoints must be enabled. In
# interactive mode, ROCgdb will ask the user if the unknown breakpoint is supposed to be pending.
set breakpoint pending on

# Temporary breakpoint in GPU kernel code - will be automatically deleted after the first encounter. Since this is a
# pending breakpoint (it is set before the GPU kernel code is loaded) ROCgdb will issue a warning:
# "Function "matrix_multiplication_kernel" not defined." In this case, it can be safely ignored.
tbreak matrix_multiplication_kernel

# In interactive sessions, pagination should be disabled for non-stop mode.
# set pagination off

# Enable non-stop mode - this must be done before the examined program is launched.
# In non-stop mode, only the current block / work-group is stopped, while the other blocks resume their work. This is
# useful if asynchronous activities should continue after the block of interest is stopped, for example to observe
# changes in global memory done by other blocks.
set non-stop on

run

# Switch to a GPU thread
thread 8

# Set breakpoint for GPU thread before global memory is written.
break 109 thread 8

# In non-stop mode, 'continue' only affects the current wavefront. To resume all threads, the '-a' parameter is
# required.
continue -a

echo -------------------------------------------------------------------------------\n
echo Inspecting output buffer\n
echo -------------------------------------------------------------------------------\n
x/128fw C

# Resume all threads.
continue -a