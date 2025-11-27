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
echo Modifying wavefront execution\n
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
echo Locking scheduler execution\n
echo -------------------------------------------------------------------------------\n
# If a GPU breakpoint is reached, 'step' can be used to step through the GPU code. By default, this still allows other
# wavefronts to progress independently of the currently observed wavefront. To prevent this, the following instruction
# can be used.
set scheduler-locking on

echo -------------------------------------------------------------------------------\n
echo Switching thread\n
echo -------------------------------------------------------------------------------\n
# After setting the scheduler lock, any other wavefront will not have progressed past the point of the currently
# examined wavefront. This way it is easily possible to switch between threads and examine their data at the same point
# in a kernel's lifetime.
thread 64

# Now the other thread's or wavefront's internals can be inspected.

echo -------------------------------------------------------------------------------\n
echo Unlocking scheduler execution\n
echo -------------------------------------------------------------------------------\n
# If scheduler-locking is enabled, 'continue' will only resume the current wavefront. To let the entire application
# continue, scheduler-locking needs to be disabled (or you must cycle through the wavefronts and continue them
# manually).
set scheduler-locking off

continue
