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

EXAMPLE := hip_pinned_host_memory
GPU_RUNTIME ?= HIP

# HIP variables
ROCM_PATH ?= /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_PATH)/include

HIPCXX ?= $(ROCM_PATH)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -isystem $(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

ifeq ($(GPU_RUNTIME), CUDA)
  ICXXFLAGS += -x cu
else ifeq ($(GPU_RUNTIME), HIP)
  CXXFLAGS  ?= -Wall -Wextra
  ICXXFLAGS += -D__HIP_PLATFORM_AMD__
  ILDFLAGS  += -L $(ROCM_PATH)/lib
  ILDLIBS   += -lamdhip64
else
  $(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be either CUDA or HIP)
endif

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

$(EXAMPLE): main.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean
