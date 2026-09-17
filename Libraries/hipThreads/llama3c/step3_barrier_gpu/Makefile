# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

EXAMPLE            := hipthreads_llama3c_step3_barrier_gpu
COMMON_INCLUDE_DIR := ../../../../Common

GPU_RUNTIME ?= HIP

ifneq ($(GPU_RUNTIME), HIP)
  $(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP.)
endif

ROCM_PATH ?= /opt/rocm

HIPCXX ?= $(ROCM_PATH)/bin/hipcc

HIP_INCLUDE_DIR        := $(ROCM_PATH)/include
HIPTHREADS_INCLUDE_DIR := $(ROCM_PATH)/include/hipthreads
LIBHIPCXX_INCLUDE_DIR  := $(ROCM_PATH)/include/libhipcxx

CXX_STD   := c++17
CXXFLAGS  ?= -Wall -Wextra
ICXXFLAGS := -std=$(CXX_STD) -O3 -fgpu-rdc $(CXXFLAGS)
ILDFLAGS  := -fgpu-rdc -L$(ROCM_PATH)/lib/hipthreads $(LDFLAGS)
ICPPFLAGS := -isystem $(HIP_INCLUDE_DIR) -isystem $(HIPTHREADS_INCLUDE_DIR) \
             -I $(LIBHIPCXX_INCLUDE_DIR) -I $(COMMON_INCLUDE_DIR) \
             $(CPPFLAGS)
ILDLIBS   := -lhipthreads $(LDLIBS)

$(EXAMPLE): llama3.cxx win.h
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) -x hip -o $@ llama3.cxx $(ILDFLAGS) $(ILDLIBS)

test: $(EXAMPLE)
	# Running requires a model file; skip automated test

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean test
