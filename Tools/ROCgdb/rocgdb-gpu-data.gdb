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
echo Examining GPU data\n
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
echo Register information\n
echo -------------------------------------------------------------------------------\n
# The following displays all registers and their contents.
info registers

echo -------------------------------------------------------------------------------\n
echo Examining vector register\n
echo -------------------------------------------------------------------------------\n
# The following will print the contents of a single vector register, i.e. all the elements held by the register.
print $v42

echo -------------------------------------------------------------------------------\n
echo Examining scalar register\n
echo -------------------------------------------------------------------------------\n
print $s42

echo -------------------------------------------------------------------------------\n
echo Examining global memory buffer\n
echo -------------------------------------------------------------------------------\n
# 'x' is the command for examining a data buffer, '8fw' are formatting arguments: The first _8_ elements, in _f_loat
# format, where each element is the size of a _w_ord (4 bytes). 'A' an input parameter to the GPU kernel.
x/8fw A

echo -------------------------------------------------------------------------------\n
echo Examining local memory buffer\n
echo -------------------------------------------------------------------------------\n
# Sometimes it is necessary to explicitly specify an address space. The following shows how to do this for a buffer in
# local (= shared) memory. Other valid address spaces are "global", "generic", "private_wave" and "private_lane". With
# explicit address space specifiers it is necessary to pass an address instead of a variable name. The address 0x0
# corresponds to the start of the kernel's __shared__ buffer 'a_values'.
x/8fw local#0x0

continue
